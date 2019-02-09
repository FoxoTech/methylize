import statsmodels.api as sm
import numpy as np
import pandas as pd

def detect_DMPs(meth_data,pheno_data,regression_method="linear",q_cutoff=1,shrink_var=False):
    """
    This function searches for individual differentially methylated positions/probes
    (DMPs) by regressing the methylation M-value for each sample at a given
    genomic location against the phenotype data for those samples.

    Phenotypes can be provided as a list of string-based or integer binary data
    or as numeric continuous data.

    Inputs and Parameters
    ---------------------------------------------------------------------------
        meth_data: (CURRENTLY) A pandas dataframe of methylation M-values for
                  where each column corresponds to a CpG site probe and each
                  row corresponds to a sample.
        pheno_data: A list or one dimensional numpy array of phenotypes
                   for each sample column of meth_data.
                   - Binary phenotypes can be presented as a list/array
                     of zeroes and ones or as a list/array of strings made up
                     of two unique words (i.e. "control" and "cancer"). The first
                     string in phenoData will be converted to zeroes, and the
                     second string encountered will be convered to ones for the
                     logistic regression analysis.
        regression_method: Either the string "logistic" or the string "linear"
                           depending on the phenotype data available. (Default:
                           "linear") Phenotypes with only two options
                           (e.g. "control" and "cancer") should be analyzed
                           with a logistic regression, whereas continuous numeric
                           phenotypes are required to run the linear regression analysis.
        q_cutoff: Select a cutoff value to return only those DMPs that meet a
                 particular significance threshold. Reported q-values are
                 p-values corrected according to the model's false discovery
                 rate (FDR). Default = 1 to return all DMPs regardless of
                 significance.
        shrink_var: If True, variance shrinkage will be employed and squeeze
                   variance using Bayes posterior means. Variance shrinkage
                   is recommended when analyzing small datasets (n < 10).
                   (NOT IMPLEMENTED YET)

    Returns:
        A pandas dataframe of regression statistics with a row for each probe analyzed
        and columns listing the individual probe's regression statistics of:
            - regression coefficient
            - lower limit of the coefficient's 95% confidence interval
            - upper limit of the coefficient's 95% confidence interval
            - standard error
            - p-value
            - q-value (p-values corrected for multiple testing using the Benjamini-Hochberg FDR method)

        The rows are sorted by q-value in ascending order to list the most significant
        probes first. If q_cutoff is specified, only probes with significant q-values
        less than the cutoff will be returned in the dataframe.
    """

    ##Check that an available regression method has been selected
    regression_options = ["logistic","linear"]
    if regression_method not in regression_options:
        raise ValueError("Either a 'linear' or 'logistic' regression must be specified for this analysis.")

    ##Check that meth_data is a numpy array with float type data
    if type(meth_data) is np.ndarray:
        if not np.issubdtype(meth_data.dtype, np.number):
            raise ValueError("Methylation values must be numeric data")
    else:
        raise ValueError("Methylation values must be in numeric numpy array")

    ##Check that the methylation and phenotype data correspond to the same number of samples
    if len(pheno_data) != meth_data.shape[1]:
        raise ValueError("Methylation data and phenotypes must have the same number of samples")

    ##Extract column names corresponding to all probes to set row indices for results
    all_probes = meth_data.columns.values.tolist()
    ##List the statistical output to be produced for each probe's regression
    stat_cols = ["Coefficient","95%CI_lower","95%CI_upper","StandardError","PValue","FDR_QValue"]
    ##Create empty pandas dataframe with probe names as row index to hold stats for each probe
    probe_stats = pd.DataFrame(index=all_probes,columns=stat_cols)
    ##Fill with NAs
    probe_stats = probe_stats.fillna()
    
    ##Run logistic regression for binary phenotype data
    if regression_method == "logistic":
        ##Check that binary phenotype data actually has 2 distinct categories
        pheno_options = set(pheno_data)
        if len(pheno_options) < 2:
            raise ValueError("Binary phenotype analysis requires 2 different phenotypes, but only 1 is detected.")
        elif len(pheno_options) > 2:
            raise ValueError("Binary phenotype analysis requires 2 different phenotypes, but more than 2 are detected.")

        ##Check if binary phenotype data is already formatted as 0's and 1's that
            ##can be coerced to integers
        try:
            int(list(pheno_options)[0])
            try:
                int(list(pheno_options)[1])
                integers = True
                if 0 in pheno_options and 1 in pheno_options:
                    zeroes_ones = True
                else:
                    zeroes_ones = False
            except:
                zeroes_ones = False
        except:
            zeroes_ones = False

        ##Format binary data as 0's and 1's if it was given as a list of strings with
            ##2 different string values
        if zeroes_ones:
            pheno_data_binary = np.array(pheno_data,dtype=np.int)
        else:
            pheno_data_binary = np.array(pheno_data)
            ##Turn the first phenotype into zeroes wherever it occurs in the array
            zero_inds = np.where(pheno_data_binary == list(pheno_options)[0])[0]
            ##Turn the second occuring phenotype into ones
            one_inds = np.where(pheno_data_binary == list(pheno_options)[1])[0]
            pheno_data_binary[zero_inds] = 0
            pheno_data_binary[one_inds] = 1
            ##Coerce array class to integers
            pheno_data_binary = np.array(pheno_data_binary,dtype=np.int)
            ##Print a message to let the user know what values were converted to zeroes and ones
            print("WARNING: Because phenotypes were provided as values other than 0 and 1, the all samples with the phenotype %s were assigned a value of 0 and all samples with the phenotype %s were assigned a value of 1 for the regression analysis." % (list(pheno_options)[0],list(pheno_options)[1]))
        
        ##Fit logistic regression for each probe of methylation data
        for probe in range(meth_data.shape[1]):
            logit = sm.Logit(pheno_data_binary,meth_data[:,probe])
            results = logit.fit()
            ##Extract desired statistical measures from logistic fit object
            probe_coef = results.params
            probe_CI = results.conf_int(0.05)  ##returns the lower and upper bounds for the coefficient's 95% confidence interval
            probe_pval = results.pvalues
            probe_SE = results.bse
            ##Fill in the corresponding row of the results dataframe with these values
            probe_stats.loc[all_probes[probe]] = {"Coefficient":probe_coef,"95%CI_lower":probe_CI[0][0],"95%CI_upper":probe_CI[0][1],"StandardError":probe_SE,"PValue":probe_pval}
        ##Correct all the p-values for multiple testing
        probe_stats["FDR_QValue"] = sm.multipletests(probe_stats["PValue"],alpha=0.05,method="fdr_bh")
        ##Sort dataframe by q-values, ascending, to list most significant probes first
        probe_stats = probe_stats.sort_values("FDR_QValue",axis=0)
        ##Limit dataframe to probes with q-values less than the specified cutoff
        probe_stats = probe_stats.loc[probe_stats["FDR_QValue"] < q_cutoff]

    ##Run OLS regression on continuous phenotype data
    elif regression_method == "linear":
        ##Check that phenotype data can be converted to a numeric array
        try:
            pheno_data_array = np.array(pheno_data,dtype="float_")
        except:
            raise ValueError("Phenotype data cannot be converted to a continuous numeric data type.")

        ##Fit least squares regression to each probe of methylation data
        for probe in range(meth_data.shape[1]):
            model = sm.OLS(meth_data[:,probe],pheno_data_array)
            results = model.fit()
            probe_coef = results.param
            probe_CI = results.conf_int(0.05)   ##returns the lower and upper bounds for the coefficient's 95% confidence interval
            probe_SE = results.bse
            probe_pvals = results.pvalues
            ##Fill in the corresponding row of the results dataframe with these values
            probe_stats.loc[all_probes[probe]] = {"Coefficient":probe_coef,"95%CI_lower":probe_CI[0][0],"95%CI_upper":probe_CI[0][1],"StandardError":probe_SE,"PValue":probe_pval}
        ##Correct all the p-values for multiple testing
        probe_stats["FDR_QValue"] = sm.multipletests(probe_stats["PValue"],alpha=0.05,method="fdr_bh")
        ##Sort dataframe by q-value, ascending, to list most significant probes first
        probe_stats = probe_stats.sort_values("FDR_QValue",axis=0)
        ##Limit dataframe to probes with q-values less than the specified cutoff
        probe_stats = probe_stats.loc[probe_stats["FDR_QValue"] < q_cutoff]

    return probe_stats


                         
                         
    
    

import statsmodels.api as sm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
import matplotlib.pyplot as plt

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
    if type(meth_data) is pd.DataFrame:
        meth_dtypes = list(set(meth_data.dtypes))
        for d in meth_dtypes:
            if not np.issubdtype(d, np.number):
                raise ValueError("Methylation values must be numeric data")
    else:
        raise ValueError("Methylation values must be in a pandas DataFrame")

    ##Check that the methylation and phenotype data correspond to the same number of samples
    if len(pheno_data) != meth_data.shape[0]:
        raise ValueError("Methylation data and phenotypes must have the same number of samples")

    ##Extract column names corresponding to all probes to set row indices for results
    all_probes = meth_data.columns.values.tolist()
    ##List the statistical output to be produced for each probe's regression
    stat_cols = ["Coefficient","StandardError","PValue","FDR_QValue","95%CI_lower","95%CI_upper"]
    ##Create empty pandas dataframe with probe names as row index to hold stats for each probe
    global probe_stats
    probe_stats = pd.DataFrame(index=all_probes,columns=stat_cols)
    ##Fill with NAs
    probe_stats = probe_stats.fillna(np.nan)
    
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
            print("Because phenotypes were provided as values other than 0 and 1, all samples with the phenotype %s were assigned a value of 0 and all samples with the phenotype %s were assigned a value of 1 for the logistic regression analysis." % (list(pheno_options)[0],list(pheno_options)[1]))
        
        ##Fit least squares regression to each probe of methylation data
            ##Parallelize across all available cores using joblib
        f = delayed(logistic_DMP_regression)
        n_jobs = cpu_count()
        
        with Parallel(n_jobs=n_jobs) as parallel:
            ##Apply the linear regression function to each column in meth_data (all use the same phenotype data array)
            probe_stat_rows = parallel(f(meth_data[x],pheno_data_binary) for x in meth_data)
            ##Concatenate the probes' statistics together into one dataframe
            logistic_probe_stats = pd.concat(probe_stat_rows,axis=1) 
        
        ##Combine the parallel-processed linear regression results into one pandas dataframe
            ##The concatenation after joblib's parallellization produced a dataframe with a column for each probe
            ##so transpose it to probes by rows instead
        probe_stats = logistic_probe_stats.T
        
        ##Pull out probes that encountered perfect separation or linear algebra errors to remove them from the 
            ##final stats dataframe while alerting the user to the issues fitting regressions to these individual probes
        perfect_sep_probes = probe_stats.index[probe_stats["PValue"]==-999]
        linalg_error_probes = probe_stats.index[probe_stats["PValue"]==-995]
        probe_stats = probe_stats.drop(index=perfect_sep_probes)
        probe_stats = probe_stats.drop(index=linalg_error_probes)
        
        ##Remove any rows that still have NAs (probes that couldn't be analyzed due to perfect separation or LinAlgError)
        probe_stats = probe_stats.dropna(axis=0,how="all")
        ##Correct all the p-values for multiple testing
        probe_stats["FDR_QValue"] = sm.stats.multipletests(probe_stats["PValue"],alpha=0.05,method="fdr_bh")[1]
        ##Sort dataframe by q-values, ascending, to list most significant probes first
        probe_stats = probe_stats.sort_values("FDR_QValue",axis=0)
        ##Limit dataframe to probes with q-values less than the specified cutoff
        probe_stats = probe_stats.loc[probe_stats["FDR_QValue"] < q_cutoff]
        
        ##Print a message to let the user know how many and which probes failed
            ##with perfect separation
        if len(perfect_sep_probes) > 0:
            print("%s probes failed the logistic regression analysis due to perfect separation and could not be included in the final results." % len(perfect_sep_probes))
            print("Probes with perfect separation errors:")
            for i in perfect_sep_probes:
                print(i)
        if len(linalg_error_probes) > 0:
            print("%s probes failed the logistic regression analysis due to encountering a LinAlgError: Singular matrix and could not be included in the final results." % len(linalg_error_probes))
            print("Probes with LinAlgError:")
            for i in linalg_error_probes:
                print(i)

    ##Run OLS regression on continuous phenotype data
    elif regression_method == "linear":
        ##Make the phenotype data a global variable
        global pheno_data_array
        ##Check that phenotype data can be converted to a numeric array
        try:
            pheno_data_array = np.array(pheno_data,dtype="float_")
        except:
            raise ValueError("Phenotype data cannot be converted to a continuous numeric data type.")

        ##Fit least squares regression to each probe of methylation data
            ##Parallelize across all available cores using joblib
        f = delayed(linear_DMP_regression)
        n_jobs = cpu_count()
        
        with Parallel(n_jobs=n_jobs) as parallel:
            ##Apply the linear regression function to each column in meth_data (all use the same phenotype data array)
            probe_stat_rows = parallel(f(meth_data[x],pheno_data_array) for x in meth_data)
            ##Concatenate the probes' statistics together into one dataframe
            linear_probe_stats = pd.concat(probe_stat_rows,axis=1) 
        
        ##Combine the parallel-processed linear regression results into one pandas dataframe
            ##The concatenation after joblib's parallellization produced a dataframe with a column for each probe
            ##so transpose it to probes by rows instead
        probe_stats = linear_probe_stats.T
        
        ##Correct all the p-values for multiple testing
        probe_stats["FDR_QValue"] = sm.stats.multipletests(probe_stats["PValue"],alpha=0.05,method="fdr_bh")[1]
        ##Sort dataframe by q-value, ascending, to list most significant probes first
        probe_stats = probe_stats.sort_values("FDR_QValue",axis=0)
        ##Limit dataframe to probes with q-values less than the specified cutoff
        probe_stats = probe_stats.loc[probe_stats["FDR_QValue"] < q_cutoff]
        ##Alert the user if there are no significant DMPs within the cutoff range they specified
        if probe_stats.shape[0] == 0:
            print("No DMPs were found within the q = %s significance cutoff level specified." %q_cutoff)
    
    ##Return a dataframe of regression statistics with a row for each probe and a column for each statistical measure
    return probe_stats

                         

def linear_DMP_regression(probe_data,phenotypes):
    """
    This function performs a linear regression on a single probe's worth of methylation
    data (in the form of M-values). It is called by the detect_DMPs.

    Inputs and Parameters
    ---------------------------------------------------------------------------
        probe_data: A pandas Series for a single probe with a methylation M-value 
                    for each sample in the analysis. The Series name corresponds 
                    to the probe ID, and the Series is extracted from the meth_data
                    DataFrame through a parallellized loop in detect_DMPs.
        phenotypes: A numpy array of numeric phenotypes with one phenotype per
                    sample (so it must be the same length as probe_data). This is 
                    the same object as the pheno_data input to detect_DMPs after
                    it has been checked for data type and converted to the
                    numpy array pheno_data_array.

    Returns:
        A pandas Series of regression statistics for the single probe analyzed.
        The columns of regression statistics are as follows:
            - regression coefficient
            - lower limit of the coefficient's 95% confidence interval
            - upper limit of the coefficient's 95% confidence interval
            - standard error
            - p-value
    """
    ##Find the probe name for the single pandas series of data contained in probe_data
    probe_ID = probe_data.name
    ##Fit OLS linear model individual probe
    model = sm.OLS(probe_data,phenotypes)
    results = model.fit()
    probe_coef = results.params
    probe_CI = results.conf_int(0.05)   ##returns the lower and upper bounds for the coefficient's 95% confidence interval
    probe_SE = results.bse
    probe_pval = results.pvalues
    ##Fill in the corresponding row of the results dataframe with these values
    probe_stats_row = pd.Series({"Coefficient":probe_coef[0],"StandardError":probe_SE[0],"PValue":probe_pval[0],"95%CI_lower":probe_CI[0][0],"95%CI_upper":probe_CI[1][0]},name=probe_ID)
    return probe_stats_row

def logistic_DMP_regression(probe_data,phenotypes):
    """
    This function performs a logistic regression on a single probe's worth of methylation
    data (in the form of M-values). It is called by the detect_DMPs.

    Inputs and Parameters
    ---------------------------------------------------------------------------
        probe_data: A pandas Series for a single probe with a methylation M-value 
                    for each sample in the analysis. The Series name corresponds 
                    to the probe ID, and the Series is extracted from the meth_data
                    DataFrame through a parallellized loop in detect_DMPs.
        phenotypes: A numpy array of binary phenotypes with one phenotype per
                    sample (so it must be the same length as probe_data). This is 
                    the same object as the pheno_data input to detect_DMPs after
                    it has been checked for data type and converted to the
                    numpy array pheno_data_binary.

    Returns:
        A pandas Series of regression statistics for the single probe analyzed.
        The columns of regression statistics are as follows:
            - regression coefficient
            - lower limit of the coefficient's 95% confidence interval
            - upper limit of the coefficient's 95% confidence interval
            - standard error
            - p-value
            
        If the logistic regression was unsuccessful in fitting to the data due
        to a Perfect Separation Error (as may be the case with small sample sizes)
        or a Linear Algebra Error, the exception will be caught and the probe_stats_row
        output will contain dummy values to flag the error. Perfect Separation Errors
        are coded with the value -999 and Linear Algebra Errors are coded with value
        -995. These rows are processed and removed in the next step of detect_DMPs to
        prevent them from interfering with the final analysis and p-value correction
        while printing a list of the unsuccessful probes to alert the user to the issues.
    """
    ##Find the probe name for the single pandas series of data contained in probe_data
    probe_ID = probe_data.name
    ##Fit the logistic model to the individual probe
    logit = sm.Logit(phenotypes,probe_data)
    try:
        results = logit.fit()
        ##Extract desired statistical measures from logistic fit object
        probe_coef = results.params
        probe_CI = results.conf_int(0.05)  ##returns the lower and upper bounds for the coefficient's 95% confidence interval
        probe_CI = np.array(probe_CI)  ##conf_int returns a pandas dataframe, easier to work with array for extracting results though
        probe_pval = results.pvalues
        probe_SE = results.bse
        ##Fill in the corresponding row of the results dataframe with these values
        probe_stats_row = pd.Series({"Coefficient":probe_coef[0],"StandardError":probe_SE[0],"PValue":probe_pval[0],"95%CI_lower":probe_CI[0][0],"95%CI_upper":probe_CI[0][1]},name=probe_ID)
    except Exception as ex:
        ##If there's a perfect separation error that prevents the model from being fit (like due to small sample sizes),
            ##add that probe name to a list to alert the user later that these probes could not be fit with a logistic regression
        if type(ex).__name__ == "PerfectSeparationError":
            probe_stats_row = pd.Series({"Coefficient":-999,"StandardError":-999,"PValue":-999,"95%CI_lower":-999,"95%CI_upper":-999},name=probe_ID)
        elif type(ex).__name__ == "LinAlgError":
            probe_stats_row = pd.Series({"Coefficient":-995,"StandardError":-995,"PValue":-995,"95%CI_lower":-995,"95%CI_upper":-995},name=probe_ID)
        else:
            raise ex
    return probe_stats_row


def volcano_plot(stats_results,cutoff=0.05):
    """
    This function writes the pandas DataFrame output of detect_DMPs to a CSV file
    named by the user. The DataFrame has a row for every successfully tested probe
    and columns with different regression statistics as follows:
            - regression coefficient
            - lower limit of the coefficient's 95% confidence interval
            - upper limit of the coefficient's 95% confidence interval
            - standard error
            - p-value
            - q-value (p-values corrected for multiple testing using the Benjamini-Hochberg FDR method)

    Inputs and Parameters
    ---------------------------------------------------------------------------
        stats_results: A pandas DataFrame output by the function detect_DMPs.
        cutoff: The significance level that will be used to highlight the most
                significant adjusted p-values (FDR Q-values) on the plot.
                (default = 0.05 alpha level)
        
    Returns:
        Displays a plot, but does not directly return an object.
        The data is color coded and displayed as follows:
            - the negative log of adjusted p-values is plotted on the y-axis
            - the regression coefficient beta value is plotted on the x-axis
            - the significance cutoff level appears as a horizontal gray dashed line
            - non-significant points appear in light gray
            - significant points with positive correlations (hypermethylated probes)
              appear in red
            - significant points with negative correlations (hypomethylated probes)
              appear in blue
    """
    colors = []
    for i in range(len(stats_results.FDR_QValue)):
        if stats_results.FDR_QValue[i] < cutoff:
            if stats_results.Coefficient[i] > 0:
                colors.append("red")
            else:
                colors.append("blue")
        else:
            colors.append("silver")
    plt.scatter(stats_results.Coefficient,-np.log10(stats_results.FDR_QValue),c=colors)
    plt.ylabel("-log10 (FDR Adjusted Q Value)")
    plt.xlabel("Beta")
    plt.axhline(y=-np.log10(cutoff), color="gray", linestyle='--')
    plt.show()


def write_results_to_CSV(stats_results,filename):
    """
    This function writes the pandas DataFrame output of detect_DMPs to a CSV file
    named by the user. The DataFrame has a row for every successfully tested probe
    and columns with different regression statistics as follows:
            - regression coefficient
            - lower limit of the coefficient's 95% confidence interval
            - upper limit of the coefficient's 95% confidence interval
            - standard error
            - p-value
            - q-value (p-values corrected for multiple testing using the Benjamini-Hochberg FDR method)

    Inputs and Parameters
    ---------------------------------------------------------------------------
        stats_results: A pandas DataFrame output by the function detect_DMPs.
        filename: A string that will be used to name the resulting .CSV file.
        
    Returns:
        Writes a CSV file, but does not directly return an object.
        The CSV will include the DataFrame column names as headers and the index
        of the DataFrame as row names for each probe.
    """
    stats_results.to_csv(filename)
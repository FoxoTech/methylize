import statsmodels.api as sm
import numpy as np

def detect_DMPs(meth_data,pheno_data,pheno_data_type,q_cutoff=1,shrink_var=False):
    """
    This function searches for individual differentially methylated positions/probes
    (DMPs) by regressing the methylation M-value for each sample at a given
    genomic location against the phenotype data for those samples.

    Phenotypes can be provided as a list of string-based or integer binary data
    or as numeric continuous data.

    Inputs and Parameters
    ---------------------------------------------------------------------------
        methData: (CURRENTLY) A numpy array of methylation M-values for
                  where each row corresponds to a CpG site probe and each
                  column corresponds to a sample.
        phenoData: A list or one dimensional numpy array of phenotypes
                   for each sample column of meth_data.
                   - Binary phenotypes can be presented as a list/array
                     of zeroes and ones or as a list/array of strings made up
                     of two unique words (i.e. "control" and "cancer"). The first
                     string in phenoData will be converted to zeroes, and the
                     second string encountered will be convered to ones.
        phenoDataType: Either the string "categorical" if the phenotypes are
                       qualitative in nature, or the string "continuous" if the
                       phenotypes are numeric measurements.
        qCutoff: Select a cutoff value to return only those DMPs that meet a
                 particular significance threshold. Reported q-values are
                 p-values corrected according to the model's false discovery
                 rate (FDR). Default = 1 to return all DMPs regardless of
                 significance.
        shrinkVar: If True, variance shrinkage will be employed and squeeze
                   variance using Bayes posterior means. Variance shrinkage
                   is recommended when analyzing small datasets (n < 10).
                   (NOT IMPLEMENTED YET)

    Returns:
        
    """

    ##Check that phenotype data has been specified as either binary or continuous
    data_types = ["binary","continuous"]
    if pheno_data_type not in data_types:
        raise ValueError("Phenotype data must be described as either 'binary' or 'continuous'")

    ##Check that meth_data is a numpy array with float type data
    if type(meth_data) is np.ndarray:
        if not np.issubdtype(meth_data.dtype, np.number):
            raise ValueError("Methylation values must be numeric data")
    else:
        raise ValueError("Methylation values must be in numeric numpy array")

    ##Check that the methylation and phenotype data correspond to the same number of samples
    if len(pheno_data) != meth_data.shape[1]:
        raise ValueError("Methylation data and phenotypes must have the same number of samples")

    ##Format methylation data for regression
    meth_data_T = np.transpose(meth_data)  ##transpose meth_data to make a row for each sample and column for each probe

    ##Run logistic regression for binary phenotype data
    if pheno_data_type == "binary":
        ##Check that binary phenotype data actually has 2 distinct categories
        pheno_options = set(pheno_data_type)
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
        
        ##Fit logistic regression to phenotype and M-value data
        logit = sm.Logit(pheno_data_binary,meth_data_T)
        ##Encountering PERFECT SEPARATION ERROR, need to reduce dimensionality
        results = logit.fit()

    ##Run OLS regression on continuous phenotype data
    else:
        ##Check that phenotype data can be converted to a numeric array
        try:
            pheno_data_array = np.array(pheno_data,dtype="float_")
        except:
            raise ValueError("Phenotype data cannot be converted to a continuous numeric data type.")
        ##Fit least squares regression to each probe of methylation data
        probe_pvals = []
        probe_coefs = []
        probe_SE = []
        for probe in range(meth_data_T.shape[1]):
            model = sm.OLS(meth_data_T[:,probe],pheno_data_array)
            results = model.fit()
            probe_coefs.append(results.params)
            probe_SE.append(results.bse)
            probe_pvals.append(results.pvalues)
        return probe_pvals,probe_coefs,probe_SE


                         
                         
    
    

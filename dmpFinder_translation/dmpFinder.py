import statsmodels.api as sm

def dmpFinder(methData,phenoData,phenoDataType,qCutoff=1,shrinkVar=False):
    """
    This function searches for individual differentially methylated positions
    (DMPs) by regressing the methylation M-value for each sample at a given
    genomic location against the phenotype data for those samples.

    Phenotypes can be provided as a list of string-based categorical data
    or as numeric continuous data.

    Inputs and Parameters
    ---------------------------------------------------------------------------
        methData: (CURRENTLY) A numpy array of methylation M-values for
                  where each row corresponds to a CpG site probe and each
                  column corresponds to a sample.
        phenoData: A list or one dimensional numpy array of phenotypes
                   for each sample column of methData.
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

    ##Check that phenotype data has been specified as either categorical or continuous
    data_types = ["categorical","continuous"]
    if phenoDataType not in data_types:
        raise ValueError("Phenotype data must be described as either 'categorical' or 'continuous'")

    ##Check that methData is a numpy array with float type data
    if type(methData) is np.array:
        if not np.issubdtype(methData.dtype, np.number):
            raise ValueError("Methylation values must be numeric data")
    else:
        raise ValueError("Methylation values must be in numeric numpy array")

    ##Check that the methylation and phenotype data correspond to the same number of samples
    if len(phenoData) != methData.shape[1]:
        raise ValueError("Methylation data and phenotypes must have the same number of samples")

    ##Fit model for categorical phenotype data
    if phenoDataType == "categorical":
        sm.OLS()
    







    



                         
                         
    
    

import logging
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.api import DescrStatsW
from scipy.stats import linregress, pearsonr, norm, sem
from scipy.stats import t as student_t
from joblib import Parallel, delayed, cpu_count
from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib # color maps
import datetime
import methylprep
from tqdm.autonotebook import tqdm

# app
from .helpers import color_schemes, create_probe_chr_map, create_mapinfo

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def diff_meth_pos(
    meth_data,
    pheno_data,
    regression_method=None,
    impute='delete',
    **kwargs):
    """
This function searches for individual differentially methylated positions/probes
(DMPs) by regressing the methylation M-value for each sample at a given
genomic location against the phenotype data for those samples.

Phenotypes can be provided as a list of string-based or integer binary data
or as numeric continuous data.

Input Parameters:

    meth_data:
        A pandas dataframe of methylation array data (as M-values)
        where each column corresponds to a CpG site probe and each
        row corresponds to a sample. IF a dataframe of beta-values is supplied instead,
        this function will detect this and convert to M-values before proceeding.
    pheno_data:
        A list or one dimensional numpy array of phenotypes for each sample row in meth_data.
        Binary phenotypes can be presented as a list/array
        of zeroes and ones or as a list/array of strings made up
        of two unique words (i.e. "control" and "cancer").
        - linear regression requires a measurement for the phenotype
        - note: methylprep creates a `sample_sheet_meta_data.pkl` file containing the phenotype
        data for this input. You just need to load it and specify which `column` to be used as the pheno_data.
        - If pheno_data is a pandas DataFrame, you must specify a `column` and may optionally
        specify `covariates` from the other columns in the DataFrame. Numpy matrices are NOT supported.
    column:
        - If pheno_data is a DataFrame, column='label' will select one series to be used as the phenotype data.
    covariates: (default None) use to define a more complex model for logistic regression.
        - if pheno_data is not a DataFrame, covariates will be ignored.
        - if pheno_data is a DataFrame, and `column` is specified, and covariates is None, it will ignore other columns in pheno_data.
        - if covariates is a list of strings, only these column names will be used as the covariate data from pheno_data.
        - If covariates is set to True, then all other columns will be treated as covariates in logistic regression.
    regression_method: (logistic | linear)
        - Either the string "logistic" or the string "linear"
        depending on the phenotype data available.
        - Phenotypes with only two options (e.g. "control" and "cancer") can be analyzed with a logistic regression
        - Continuous numeric phenotypes (e.g. age) are required to run a linear regression analysis.
        - Default: auto-determine, using the data types found in `pheno_data`.
        If it is numeric with more than 2 different labels, use "linear."
    impute:
        Default: 'delete' probes if ANY samples have missing data for that probe.
        True or 'auto': if <30 samples, deletes rows; if >=30 samples, uses average.
        False: don't impute and throw an error if NaNs present
        'average' - use the average of probe values in this batch
        'delete' - drop probes if NaNs are present in any sample
        'fast' - use adjacent sample probe value instead of average (much faster but less precise)
    alpha: float
        Default is 0.05 for all tests where it applies.
    fwer: float
        Set the familywise error rate (FWER). Default is 0.05, meaning that we expect 5% of all significant differences to be false positives. 0.1 (10%) is a more conservative convention for FWER.
    q_cutoff:
        - Select a cutoff value (< 1.0) to return filtered results.
        Only those DMPs that meet a particular significance threshold (e.g. 0.1) will be retained.
        Reported q-values are p-values corrected according to the model's false discovery
        rate (FDR).
        - Default: 1 -- returns all DMPs regardless of significance.
    export:
        - default: False
        - if True or 'csv', saves a csv file with data
        - if 'pkl', saves a pickle file of the results as a dataframe.
        - Use q_cutoff to limit what gets saved to only significant results.
            by default, `q_cutoff == 1` and this means everything is saved/reported/exported.
    filename:
        - specify a filename for the exported file.
        By default, if not specified, filename will be `DMP_<number of probes in file>_<number of samples processed>_<current_date>.<pkl|csv>`
    max_workers:
        (=INT) By default, this will parallelize probe processing, using all available cores.
        During testing, or when running in a virtual environment like circleci or docker or lambda, the number of available cores
        is fewer than the system's reported CPU cores, and it breaks. Use this to limit the available cores
        to some arbitrary number for testing or containerized-usage.
    debug:
        Default: False -- True turns on extra messages and runs the 'solver' in serial mode,
        disabling parallel processing of probes. This is slower but provides more detail for debugging code.
    solver:
        You can force it to use a different implementation of regression, for debugging.
        Options include:
        - 'statsmodels_OLS'
        - 'linregress' # from scipy
        - [logit_DMP() is the default, based on statsmodels.api.Logit()]
        - 'statsmodels_GLM' # for logistic regression

Returns:

    A pandas dataframe of regression statistics with a row for each probe analyzed
    and columns listing the individual probe's regression statistics of:
        - regression coefficient
        - lower limit of the coefficient's 95% confidence interval
        - upper limit of the coefficient's 95% confidence interval
        - standard error
        - p-value (phenotype group A vs B - likelihood that the difference is significant for this probe/location)
        - FDR_QValue: p-values corrected for multiple comparisons using the Benjamini-Hochberg FDR method

    The rows are sorted by q-value in ascending order to list the most significant
    probes first. If q_cutoff is specified, only probes with significant q-values
    less than the cutoff will be returned in the dataframe.


.. note::
    Imputation: because methylprep output contains missing values by default, this function requires user to either delete or impute missing values.
    - This can be disabled, and it will throw a warning if NaNs present.
    - default is to drop probes with NaNs.
    - auto:
      - If there are less than 30 samples, it will delete the missing rows.
      - If there are >= 30 samples in the batch analyzed, it will replace NaNs with the
        average value for that probe across all samples.
    - User may specify: True, 'auto', False, 'delete', 'average'

    If you get a `RuntimeError: main thread is not in main loop` error running
      `diff_meth_pos`, add `debug=True` to your function inputs. This error occurs
      in some command line environments when running this function repeatedly, and
      the parallel processing steps do not all close. Using debug mode forces it
      to analyze the probes serially, not parallel. It is a little slower but will
      always run without error under these conditions.

    If Progress Bar Missing:
      if you don't see a progress bar in your jupyterlab notebook, try this:
      - conda install -c conda-forge nodejs
      - jupyter labextension install @jupyter-widgets/jupyterlab-manager

.. todo::
    `shrink_var` feature: variance shrinkage / squeeze variance using Bayes posterior means.
    Variance shrinkage is recommended when analyzing small datasets (n < 10).
    Paper ref: http://www.uvm.edu/~rsingle/JournalClub/papers/Smyth-SAGMB-2004_eBayes+microarray.pdf
    [NOT IMPLEMENTED YET]
    """
    import warnings
    np.seterr(divide='ignore', over='ignore') # log10(0.0) happens
    warnings.filterwarnings("ignore") # exp(x) overflow error approximates to x=0

    allowed = ['verbose', 'alpha', 'fwer', 'column', 'covariates', 'solver', 'max_workers', 'debug',
    'export', 'filename', 'q_cutoff']
    if any([k for k in kwargs if k not in allowed]):
        misspelled = ', '.join([k for k in kwargs if k not in allowed])
        raise KeyError(f"Unrecognized agument(s): {misspelled}")

    verbose = False if kwargs.get('verbose') == False else True
    alpha = kwargs.get('alpha',0.05)
    fwer = kwargs.get('fwer',0.05)
    q_cutoff = kwargs.get('q_cutoff',1)


    # Check if pheno_data is a list, series, or dataframe
    covariate_data = None
    if isinstance(pheno_data, pd.DataFrame) and kwargs.get('column'):
        try:
            if kwargs.get('covariates') == True:
                covariate_data = pheno_data.drop(kwargs.get('column'), axis=1) #all but one
            elif isinstance(kwargs.get('covariates'), (list,tuple)):
                try:
                    covariate_data = pheno_data[[ covariates ]]
                except KeyError:
                    raise KeyError(f"Your covariates list of labels must match your pheno_data columns exactly.")
            elif kwargs.get('covariates') == None:
                pass # ignore extra columns in pheno_data (default)
            pheno_data = pheno_data[kwargs.get('column')]
        except Exception as e:
            raise ValueError("Column name you specified for pheno_data did not work: {kwargs.get('column')} ERROR: {e}")
    elif isinstance(pheno_data, pd.DataFrame) and pheno_data.shape[1] == 1:
        pheno_data = pheno_data[ pheno_data.columns[0] ] # convert a single-Column DataFrame to Series
    elif isinstance(pheno_data, pd.DataFrame):
        raise ValueError("You must specify a column by name when passing in a DataFrame for pheno_data.")

    # determine regression method, if not specified
    if not regression_method:
        if len(pd.Series(pheno_data).unique()) < 2:
            raise ValueError(f"pheno_data must have at least 2 distinct values")
        #allowed_dtypes = ('int64', 'Int64', 'float64', bool, 'datetime64', 'category', 'timedelta[ns]')
        allowed_dtypes = (np.int64, np.float64, bool, np.float32, np.int32)
        # NOT SURE if all np.generic objects will work like a list.
        if isinstance(pheno_data, (list, tuple, pd.Series, np.ndarray, np.generic)):
            if pd.Series(pheno_data).dtype == 'O' and ( all(pd.Series(pheno_data).str.isdigit())
                or all([isinstance(i, (int,float)) for i in pd.Series(pheno_data)])):
                # pheno IS or CAN BE converted to numeric
                if len(pd.Series(pheno_data).unique()) == 2:
                    regression_method = 'logistic'
                    # a linear regression with only 2 unique values isn't any good to begin with.
                else:
                    regression_method = 'linear'
            elif pd.Series(pheno_data).dtype == 'O' and len(pd.Series(pheno_data).unique()) == 2:
                regression_method = 'logistic' # strings
            elif pd.Series(pheno_data).dtype in allowed_dtypes and len(pd.Series(pheno_data).unique()) == 2:
                regression_method = 'logistic'
            elif pd.Series(pheno_data).dtype in allowed_dtypes and len(pd.Series(pheno_data).unique()) > 2:
                regression_method = 'linear'
            else:
                raise ValueError("Could not understand your pheno_data.")
        else:
            raise ValueError(f"pheno_data must be list-like, or if a DataFrame, specify the 'column' to use.")

    # Check that an available regression method has been selected
    regression_options = ["logistic","linear"]
    if regression_method not in regression_options:
        raise ValueError("Either 'linear' or 'logistic' regression must be specified for this analysis.")

    # Check that meth_data is a numpy array with float type data
    if type(meth_data) is pd.DataFrame:
        meth_dtypes = list(set(meth_data.dtypes))
        for d in meth_dtypes:
            if not np.issubdtype(d, np.number):
                raise ValueError("Methylation values must be numeric data")
    else:
        raise ValueError("Methylation values must be in a pandas DataFrame")

    # Check that numbers are not float16
    if any(meth_data.dtypes == 'float16'):
        raise ValueError("Convert your numbers from float16 to float32 first.")

    # Check that meth_data has probes in colummns, and transpose if necessary.
    if meth_data.shape[1] < 27000 and meth_data.shape[0] > 27000:
        meth_data = meth_data.transpose()
        LOGGER.debug(f"Your meth_data was transposed: {meth_data.shape}")
    # for case where meth has <27000 probes and only the OTHER axis matches phenotype length.
    if len(pheno_data) != meth_data.shape[0] and len(pheno_data) != meth_data.shape[1]:
        raise ValueError("Phenotype count does not match the sample count in meth_data")
    if len(pheno_data) != meth_data.shape[0] and len(pheno_data) == meth_data.shape[1]:
        meth_data = meth_data.transpose()
        LOGGER.debug(f"Your meth_data was transposed: {meth_data.shape}")

    # Check for missing values, and impute if necessary
    if any(meth_data.isna().sum()):
        if impute == 'fast':
            meth_data = meth_data.fillna(axis='index', method='bfill') # uses adjacent probe value from same sample to make MDS work.
            meth_data = meth_data.fillna(axis='index', method='ffill') # uses adjacent probe value from same sample to make MDS work.
            still_nan = int(meth_data.isna().sum(axis=1).mean())
            meth_data = meth_data.dropna(how='any', axis='columns') # <--- drops any probes missing everywhere, because probes are in columns
            LOGGER.warning(f"Dropped {still_nan} probes per sample that could not be imputed, leaving {meth_data.shape[1]} probes.")
        elif (impute == 'average' or
            (impute == 'auto' and meth_data.shape[0] >= 30) or
            (isinstance(impute,bool) and impute == True and meth_data.shape[0] >= 30)):
            pre_count = meth_data.shape[1]
            meth_data = meth_data.dropna(how='all', axis='columns') # if probe is NaN on all samples, omit.
            post_count = meth_data.shape[1]
            nan_per_sample = int(meth_data.isna().sum(axis=1).mean())
            probe_means = meth_data.mean(axis='columns')
            meth_data = meth_data.transpose().fillna(value=probe_means).transpose()
            if pre_count > post_count:
                LOGGER.warning(f"Dropped {pre_count-post_count} probes that were missing in all samples. Imputed {nan_per_sample} probes per sample.")
        elif (impute == 'delete' or
            (impute == 'auto' and meth_data.shape[0] < 30) or
            (isinstance(impute,bool) and impute in ('auto',True) and meth_data.shape[0] < 30)):
            pre_count = meth_data.shape[1]
            meth_data = meth_data.dropna(how='any', axis='columns') # if probe is NaN on ANY samples, omit that probe
            post_count = meth_data.shape[1]
            if pre_count > post_count:
                LOGGER.warning(f"Dropped {pre_count-post_count} probes with missing values from any samples")
        elif isinstance(impute,bool) and impute == False:
            raise ValueError("meth_data contains missing values, but impute step was explicitly disabled.")
        else:
            raise ValueError(f"Unrecognized impute method '{impute}': Choose from (auto, True, False, fast, average, delete)")

        if meth_data.shape[0] == 0 or meth_data.shape[1] == 0:
            raise ValueError(f"Impute method ({impute}) eliminated all probes. Cannot proceed.")

    # check if meth_data is beta or M-values, then convert to M-values if necessary
    m_values = True if ((meth_data < 0).any().sum() > 0 or (meth_data > 1).any().sum() > 0) else False
    if not m_values:
        def beta2m(val):
            return math.log2(val/(1-val))
        meth_data = meth_data.apply(np.vectorize(beta2m))
        if verbose: LOGGER.info(f"Converted your beta values into M-values; {meth_data.shape}")

    # Check that the methylation and phenotype data correspond to the same number of samples; flip if necessary
    if len(pheno_data) != meth_data.shape[0] and len(pheno_data) != meth_data.shape[1]:
        raise ValueError(f"Methylation data and phenotypes must have the same number of samples; found {len(meth_data)} meth and {len(pheno_data)} pheno.")

    ##Extract column names corresponding to all probes to set row indices for results
    all_probes = meth_data.columns.values.tolist()
    ##List the statistical output to be produced for each probe's regression
    stat_cols = ["Coefficient","StandardError","PValue","FDR_QValue","95%CI_lower","95%CI_upper"]
    ##Create empty pandas dataframe with probe names as row index to hold stats for each probe
    global probe_stats
    probe_stats = pd.DataFrame(index=all_probes,columns=stat_cols)
    ##Fill with NAs
    probe_stats = probe_stats.fillna(np.nan)


    # Run OLS regression on continuous phenotype data
    if regression_method == "linear":
        # Make the phenotype data a global variable
        global pheno_data_array
        # Check that phenotype data can be converted to a numeric array
        try:
            pheno_data_array = np.array(pheno_data, dtype="float_")
        except:
            raise ValueError("Phenotype data cannot be converted to a continuous numeric data type. Linear regression is only intended for continuous variables.")

        ##Fit least squares regression to each probe of methylation data
            ##Parallelize across all available cores using joblib
        if kwargs.get('solver') == 'statsmodels_OLS':
            func = legacy_OLS
        else:
            func = linear_DMP_regression
        n_jobs = cpu_count()
        if kwargs.get('max_workers'):
            n_jobs = int(kwargs['max_workers'])

        if kwargs.get('debug') == True:
            print(f"DEBUG MODE using linear_DMP_regression kwargs: {kwargs}")
            probe_stats_rows = []
            for x in tqdm(meth_data, total=len(all_probes), desc='Probes'):
                 probe_stats_row = func(meth_data[x], pheno_data_array, alpha=alpha)
                 # both functions gave identical slope, intercept, pvalues (2022-02-25) but StandardError differed.
                 probe_stats_rows.append(probe_stats_row)
            print('Data processing done!')
            linear_probe_stats = pd.concat(probe_stats_rows, axis=1)
        else:
            func = delayed(func)
            with Parallel(n_jobs=n_jobs) as parallel:
                parallel_cleaned_list = []
                multi_probe_errors = 0
                # para_gen() generates all the data without loading into memory, and fixes mouse array redundancy
                def para_gen(meth_data):
                    for _probe in meth_data:
                        probe_data = meth_data[_probe]
                        if isinstance(probe_data, pd.DataFrame):
                            # happens with mouse when multiple probes have the same name
                            probe_data = probe_data.mean(axis='columns')
                            probe_data.name = _probe
                            multi_probe_errors += 1
                        # columns are probes, so each probe passes in parallel
                        yield probe_data
                # Apply the linear regression function to each column in meth_data (all use the same phenotype data array)
                probe_stat_rows = parallel(func(probe_data, pheno_data_array, alpha=alpha) for probe_data in tqdm(para_gen(meth_data), total=len(all_probes), desc='Probes') )
                # Concatenate the probes' statistics together into one dataframe
                linear_probe_stats = pd.concat(probe_stat_rows, axis=1)

        # Combine the parallel-processed linear regression results into one pandas dataframe
        # The concatenation after joblib's parallellization produced a dataframe with a column for each probe
        # so transpose it to probes by rows instead
        probe_stats = linear_probe_stats.T

        """ Note:
            This function uses the False Discovery Rate (FDR) approach.
            The Benjamini–Hochberg method controls the False Discovery Rate (FDR) using sequential modified Bonferroni correction
            for multiple hypothesis testing. While the Bonferroni correction relies on the Family Wise Error Rate (FWER),
            Benjamini and Hochberg introduced the idea of a FDR to control for multiple hypotheses testing. In the statistical
            context, discovery refers to the rejection of a hypothesis. Therefore, a false discovery is an incorrect rejection
            of a hypothesis and the FDR is the likelihood such a rejection occurs. Controlling the FDR instead of the FWER is
            less stringent and increases the method’s power. As a result, more hypotheses may be rejected and more discoveries
            may be made. (From the Encyclopedia of Systems Biology: https://link.springer.com/referenceworkentry/10.1007%2F978-1-4419-9863-7_1215)
        """

        # Correct all the p-values for multiple testing
        probe_stats["FDR_QValue"] = sm.stats.multipletests(probe_stats["PValue"], alpha=fwer, method="fdr_bh")[1]
        # Sort dataframe by q-value, ascending, to list most significant probes first
        probe_stats = probe_stats.sort_values("FDR_QValue", axis=0)
        # Limit dataframe to probes with q-values less than the specified cutoff
        probe_stats = probe_stats.loc[probe_stats["FDR_QValue"] < q_cutoff]
        # Alert the user if there are no significant DMPs within the cutoff range they specified
        if probe_stats.shape[0] == 0:
            print(f"No DMPs were found within the q < {q_cutoff} (the significance cutoff level specified).")

    elif regression_method == "logistic":
        # Check if binary phenotype data is already formatted as 0's and 1's that can be coerced to integers
        try:
            pheno_options = set(list(pheno_data))
            if len(pheno_options) != 2:
                raise ValueError("Must have exactly TWO groups for logistic regression. Found {pheno_options}")
            int(list(pheno_options)[0]) # loosely allows anything that can be an INT to become INT.
            int(list(pheno_options)[1])
            integers = True
            if 0 in pheno_options and 1 in pheno_options:
                zeroes_ones = True
            else:
                zeroes_ones = False
        except:
            zeroes_ones = False

        # Format binary data as 0's and 1's if it was given as a list of strings with 2 different string values
        if zeroes_ones:
            pheno_data_binary = np.array(pheno_data,dtype=int)
        else:
            pheno_data_binary = np.array(pheno_data)
            ##Turn the first phenotype into zeroes wherever it occurs in the array
            zero_inds = np.where(pheno_data_binary == list(pheno_options)[0])[0]
            ##Turn the second occuring phenotype into ones
            one_inds = np.where(pheno_data_binary == list(pheno_options)[1])[0]
            pheno_data_binary[zero_inds] = 0
            pheno_data_binary[one_inds] = 1
            ##Coerce array class to integers
            pheno_data_binary = np.array(pheno_data_binary,dtype=int)
            ##Print a message to let the user know what values were converted to zeroes and ones
            if verbose: LOGGER.info(f"Logistic regression: Phenotype ({list(pheno_options)[0]}) was assigned to 0 and ({list(pheno_options)[1]}) was assigned to 1.")

        # Fit least squares regression to each probe of methylation data --> Parallel + joblib
        n_jobs = cpu_count()
        if kwargs.get('max_workers'):
            n_jobs = int(kwargs['max_workers'])
        func = logistic_DMP_regression if kwargs.get('solver') == 'statsmodels_GLM' else logit_DMP

        if kwargs.get('debug') == True:
            print(f'DEBUG MODE - logistic regression, kwargs: {kwargs} serial mode.')
            probe_stats_rows = []
            for probe in tqdm(list(meth_data.columns), total=len(meth_data.columns), desc='Probes'):
                probe_stats_row = func(meth_data[probe], pheno_data_binary, covariate_data=covariate_data, debug=kwargs.get('debug'))
                probe_stats_rows.append(probe_stats_row)
            print('Data processing done!')
            logistic_probe_stats = pd.concat(probe_stats_rows, axis=1)
        else:
            func = delayed(func)
            with Parallel(n_jobs=n_jobs) as parallel:
                # Apply the logistic/linear regression function to each column in meth_data (all use the same phenotype data array)
                parallel_cleaned_list = []
                multi_probe_errors = 0
                def para_gen(meth_data):
                    for _probe in meth_data:
                        probe_data = meth_data[_probe]
                        if isinstance(probe_data, pd.DataFrame):
                            # happens with mouse when multiple probes have the same name
                            probe_data = probe_data.mean(axis='columns')
                            probe_data.name = _probe
                            multi_probe_errors += 1
                        # columns are probes, so each probe passes in parallel
                        yield probe_data
                # this generates all the data without loading into memory, and fixes mouse array
                probe_stat_rows = parallel(func(probe_data, pheno_data_binary, covariate_data=covariate_data) for probe_data in tqdm(para_gen(meth_data), total=len(all_probes), desc='Probes') )
                # Concatenate the probes' statistics together into one dataframe
                logistic_probe_stats = pd.concat(probe_stat_rows, axis=1)

        # Combine the parallel-processed linear regression results into one pandas dataframe
        # The concatenation after joblib's parallellization produced a dataframe with a column for each probe
        # so transpose it to probes by rows instead
        probe_stats = logistic_probe_stats.T

        # Pull out probes that encountered perfect separation or linear algebra errors to remove them from the
        # final stats dataframe while alerting the user to the issues fitting regressions to these individual probes
        probe_stats['fold_change'] = probe_stats['fold_change'].replace(np.inf, 10)
        probe_stats['fold_change'] = probe_stats['fold_change'].replace(-np.inf, -10)
        perfect_sep_probes = probe_stats.index[probe_stats["PValue"]==-999]
        linalg_error_probes = probe_stats.index[probe_stats["PValue"]==-995]
        singular_matrix_probes = probe_stats.index[probe_stats["PValue"]==-996]
        probe_stats = probe_stats.drop(index=perfect_sep_probes)
        probe_stats = probe_stats.drop(index=linalg_error_probes)
        probe_stats = probe_stats.drop(index=singular_matrix_probes)
        unexplained_failures = list(probe_stats[ probe_stats.PValue.isna() ].index)
        # Remove any rows that still have NAs (probes that couldn't be analyzed due to perfect separation or LinAlgError)
        probe_stats = probe_stats.dropna(axis='index', how="any") # changed from 'all' -- so that ANY NaN will be dropped
        if len(probe_stats) == 0:
            LOGGER.error(f"No probes remain after filtering failed regressions:\n(perfect separation {len(perfect_sep_probes)}, Linear Algebra Error {len(linalg_error_probes)}, Singular Matrix {len(singular_matrix_probes)}, Unexplained {len(unexplained_failures)})")
            return probe_stats

        # Correct all the p-values for multiple testing
        corrections = sm.stats.multipletests(probe_stats["PValue"], alpha=fwer, method="fdr_bh")
        probe_stats["FDR_QValue"] = corrections[1]
        # Sort dataframe by q-values, ascending, to list most significant probes first
        #if len(probe_stats['FDR_QValue'].value_counts()) == 1 and len(probe_stats.loc[(probe_stats['FDR_QValue'] == 1)]) == len(probe_stats):
        #    LOGGER.warning("All probes have a p-value significance of 1.0, so your grouping variables are not reliable.")

        probe_stats = probe_stats.sort_values("FDR_QValue", axis=0)
        # Limit dataframe to probes with q-values less than the specified cutoff
        # probe_stats = probe_stats.loc[probe_stats["FDR_QValue"] <= q_cutoff]

        # Print a message to let the user know how many and which probes failed
        for fail_text, fail_reason in {'perfect separation': perfect_sep_probes,
            'LinearAlgebra error': linalg_error_probes,
            'singular matrix': singular_matrix_probes,
            'other unexplained reasons': unexplained_failures}.items():
            if len(fail_reason) > 0:
                print(f"{len(fail_reason)} probes failed the logistic regression analysis due to {fail_text} and could not be included in the final results.")
                if len(fail_reason) < 50:
                    print("Error Probes:")
                    for i in fail_reason:
                        print(i)
                elif len(fail_reason) < 100:
                    print(f"Error Probes: {fail_reason}")
        if (len(linalg_error_probes) + len(perfect_sep_probes) + len(singular_matrix_probes))/len(logistic_probe_stats.T) > 0.3:
            print("""Linear Algebra and Perfect Separation errors occur when the dataset is (a) too small to observe
events with low probabilities, or has (b) too many covariates in the model, leading to individual cell
sizes that are too small. Singular Matrix Errors occur when there is no variance in the predictors
(probe values and covariates).""")

    # Return
    if kwargs.get('export'):
        filename = kwargs.get('filename', f"DMP_{len(probe_stats)}_{len(meth_data)}_{str(datetime.date.today())}")
        if str(kwargs.get('export')).lower() == 'csv' or kwargs.get('export') == True:
            probe_stats.to_csv(filename+'.csv')
        if str(kwargs.get('export')).lower() == 'pkl':
            probe_stats.to_pickle(filename+'.pkl')
        if verbose == True:
            print(f"Saved {filename}.")
    # a dataframe of regression statistics with a row for each probe and a column for each statistical measure
    return probe_stats


def legacy_OLS(probe_data, phenotypes, alpha=0.05):  # pragma: no cover
    """ to use this, specify "statsmodels_OLS" in kwargs to diff_meth_pos()
    -- this method gives the same result as the scipy.linregress method when tested in version 1.0.0"""
    probe_ID = probe_data.name
    results = sm.OLS(probe_data, sm.add_constant(phenotypes), hasconst=True).fit()
    probe_coef = results.params.x1
    probe_CI = results.conf_int(0.05)
    probe_SE = results.bse
    probe_pval = results.f_pvalue # .pvalues are not for the fitted model
    probe_stats_row = pd.Series({
        "Coefficient":probe_coef,
        "StandardError":probe_SE.x1,
        "PValue":probe_pval,
        "95%CI_lower":probe_CI[0].x1, # probe_CI[0][0],
        "95%CI_upper":probe_CI[1].x1, # probe_CI[1][0],
        "Rsquared": results.rsquared},
        name=probe_ID)
    return probe_stats_row


def linear_DMP_regression(probe_data, phenotypes, alpha=0.05):
    """
This function performs a linear regression on a single probe's worth of methylation
data (in the form of beta or M-values). It is called by the diff_meth_pos().

Inputs and Parameters:

    probe_data:
        A pandas Series for a single probe with a methylation M-value/beta-value
        for each sample in the analysis. The Series name corresponds
        to the probe ID, and the Series is extracted from the meth_data
        DataFrame through a parallellized loop in diff_meth_pos().
    phenotypes:
        A numpy array of numeric phenotypes with one phenotype per
        sample (so it must be the same length as probe_data). This is
        the same object as the pheno_data input to diff_meth_pos() after
        it has been checked for data type and converted to the
        numpy array pheno_data_array.

Returns:

    A pandas Series of regression statistics for the single probe analyzed.
    The columns of regression statistics are as follows:
        - regression coefficient (linregress pearson's 'r')
        - lower limit of the coefficient's 95% confidence interval
        - upper limit of the coefficient's 95% confidence interval
        - standard error
        - p-value
    """
    ##Find the probe name for the single pandas series of data contained in probe_data
    probe_ID = probe_data.name
    results = linregress(phenotypes, probe_data)

    # add in confidence intervals
    r_z_value = np.arctanh(results.rvalue)
    z_score = norm.ppf(1-alpha/2)
    # t_score = t.ppf(1-alpha/2, n_samples)
    lo_z = r_z_value - (z_score * results.stderr)
    hi_z = r_z_value + (z_score * results.stderr)
    ci_lower, ci_upper = np.tanh((lo_z, hi_z))

    probe_stats_row = pd.Series({
        "Coefficient": results.slope, #results.intercept,
        "StandardError":results.stderr, # there's also results.intercept_stderr -- is this the right one?
        "PValue":results.pvalue,
        "95%CI_lower":ci_lower,
        "95%CI_upper":ci_upper,
        "Rsquared": results.rvalue**2, # rvalue is pearson's correlation r (0 to 1.0) -- square it for r-squared
    }, name=probe_ID)
    """ the OLS function gave weird results, so switched to linregress in version 1.0.0
    # adding the constant term: takes care of the bias in the data (a constant difference which is there for all observations).
    phenotypes = sm.add_constant(phenotypes)
    model = sm.OLS(probe_data, phenotypes, missing='drop')
    results = model.fit()
    probe_coef = results.params
    probe_CI = results.conf_int(alpha=0.05)   ##returns the lower and upper bounds for the coefficient's 95% confidence interval
    probe_SE = results.bse
    probe_pval = results.pvalues
    # note -- results.summary() gives a nice report on each probe, but lots of text
    ##Fill in the corresponding row of the results dataframe with these values
    probe_stats_row = pd.Series({"Coefficient":probe_coef[0], "StandardError":probe_SE[0], "PValue":probe_pval[0], "95%CI_lower":probe_CI[0][0], "95%CI_upper":probe_CI[1][0]}, name=probe_ID)
    """
    return probe_stats_row


def logit_DMP(probe_data, phenotypes, covariate_data=None, debug=False):
    """ DEFAULT method, because tested and works.
    uses statsmodels.api.Logit
    pass in a Series for probe data without the constant added

    fold_change is log2( (pheno1.mean / pheno0.mean) )

    each covariate will be normalized (to 0...1 range) before running DMP."""
    import warnings
    np.seterr(divide='ignore', over='ignore', invalid='ignore') # log10(0.0) happens
    warnings.filterwarnings("ignore") # exp(x) overflow error approximates to x=0
    probe_ID = probe_data.name
    # look at https://github.com/cozygene/glint/blob/master/utils/regression.py
    # uses statsmodels.Logit instead of statsmodels.GLM in EWAS
    #        phenotypes is n X 1 (1s or 0s)
    #        probe_data is n X 1 (the feature being tested, M-value)
    if isinstance(probe_data, pd.Series):
        probe_data = np.array(probe_data)
    if probe_data.ndim == 1:
        probe_data = probe_data.reshape(-1,1) # make sure dim is (n,1) and not(n,)
    if phenotypes.ndim == 1:
        phenotypes = phenotypes.reshape(-1, 1)

    ### confirm shape is correct here ###

    try: # log2 of ratio of group means
        # M-values are ALREADY log2 transformed, so just use the straight up difference (effect size)
        non_neg = np.array(list((val - probe_data.min())/(probe_data.max()-probe_data.min()) for val in probe_data))
        fold_change = np.log2(non_neg[ phenotypes == 1 ].mean() / non_neg[ phenotypes == 0 ].mean())
        # M-values are ALREADY log2 transformed, so just use the straight up difference (effect size)
        #fold_change = probe_data[ phenotypes == 1 ].mean() / probe_data[ phenotypes == 0 ].mean()
    except ZeroDivisionError as e:
        fold_change = np.log2(non_neg[ phenotypes == 1 ].mean() / 0.001)

    def calc_confidence_interval(data):
        """ calculates a 95% confidence interval (x  +/-  t * (s/√n))
        where data is a series of sample values for one probe."""
        (CI_lower, CI_upper) = student_t.interval(
            alpha=0.95, # e.g. 95%
            df=len(data)-1, # degrees of freedom, n-1
            loc=np.mean(data), # sample mean
            scale= sem(data)) # standard error (stdev / sqrt-of-N)
        return (CI_lower, CI_upper)
    CI_lower, CI_upper = calc_confidence_interval(probe_data)

    ### ADD INTERCEPT AFTER probes ###
    probe_data = np.insert(probe_data, 1, np.ones(len(probe_data)), axis=1)
    ### add covariates into the probe_data AFTER constant ###
    if isinstance(covariate_data, type(None)):
        pass
    elif isinstance(covariate_data, pd.DataFrame):
        # first, encode strings as ints for Logit.
        for col in covariate_data.columns:
            if covariate_data[col].dtype == 'O':
                covariate_data[col] = covariate_data[col].astype('category').cat.codes
                # normalize codes
                covariate_data[col] = covariate_data[col].apply( lambda x: x/len(covariate_data[col].unique()) )
            else:
                try:
                    # normalize, or, if every value is the same, like a constant with zero range, just set to 1.0
                    cmin = covariate_data[col].min()
                    cmax = covariate_data[col].max()
                    if cmax > cmin:
                        covariate_data[col] = (covariate_data[col] - cmin)/(cmax-cmin)
                    else:
                        covariate_data[col] = 1.0 # blanking a zero-variance column
                except Exception as e:
                    raise Exception(f"Encountered a covariate that could not be normalized: {e}\n{covariate_data[col]}")
        probe_data = np.column_stack((probe_data, covariate_data))
        # probe data is 2nd term; move to front, then all covars, then constant (1.0)
        #covar_idx = list(range(len(probe_data[0])))
        #covar_idx.remove(1) # the probe values MUST be first term.
        #covar_idx = [1] + covar_idx
        #probe_data = probe_data[:, covar_idx]
    elif not isinstance(covariate_data, pd.DataFrame):
        raise Exception(f"covariate data is not a DataFrame")

    logit_model = sm.Logit(phenotypes, probe_data) # NOTE sm.add_constant(probe_data) did NOT add col to numpy array
    try:
        results = logit_model.fit(disp=False, warn_convergence=False)
        # probe_CI = results.conf_int(0.05)  ##returns the lower and upper bounds for the coefficient's 95% confidence interval
        # probe_CI = np.array(probe_CI)  ##conf_int returns a pandas dataframe, easier to work with array for extracting results though
        ##Fill in the corresponding row of the results dataframe with these values
        probe_stats_row = pd.Series({
                'Coefficient': results.params[0],
                'PValue': results.pvalues[0], # slope
                'StandardError': results.bse[0], # slope
                'fold_change': fold_change, # effect size, assuming M-values are input == already log2 transformed betas
                "95%CI_lower": round(CI_lower[0],3),
                "95%CI_upper": round(CI_upper[0],3),
                # 'slope-t': results.tvalues[0],
                #'intercept': results.params[1],
                #'intercept-t': results.tvalues[1],
                #'intercept-p': results.pvalues[1],
                #'intercept-sem': results.bse[1],
                #'confidence': results.conf_int(0.05), not avail directly thru SKLEARN
               }, name=probe_ID)
    except Exception as e:
        fields = ['Coefficient','PValue','StandardError', 'fold_change',
            # 'slope-t', 'intercept', 'intercept-t', 'intercept-p', 'intercept-sem', 'confidence',
            "95%CI_lower", "95%CI_upper"
            ]
        # If there's a perfect separation error that prevents the model from being fit (like due to small sample sizes),
        # add that probe name to a list to alert the user later that these probes could not be fit with a logistic regression

        # CATCH Warning: invalid value encountered in sqrt
        # Warning: invalid value encountered in true_divide

        if   type(e).__name__ == "PerfectSeparationError":
            probe_stats_row = pd.Series({k:-999 for k in fields}, name=probe_ID)
        elif type(e).__name__ == "LinAlgError":
            probe_stats_row = pd.Series({k:-995 for k in fields}, name=probe_ID)
        elif type(e).__name__ == 'Singular matrix':
            probe_stats_row = pd.Series({k:-996 for k in fields}, name=probe_ID)
        elif type(e) == IndexError:
            # results are incomplete when all probe values are identical (no separation at all, so log(0)?)
            probe_stats_row = pd.Series({k:-997 for k in fields}, name=probe_ID)
        else:
            import traceback;traceback.format_exc()
            import pdb;pdb.set_trace()
            raise e
    return probe_stats_row


def logistic_DMP_regression(probe_data, phenotypes, covariate_data=None, debug=False):
    """
- Runs parallelized.
- TESTED, and gives almost the same values as logit_DMP() (2022-02-25)
- This function performs a logistic regression on a single probe's worth of methylation
data (in the form of beta/M-values). It is called by the diff_meth_pos().

Inputs and Parameters:

    probe_data:
        A pandas Series for a single probe with a methylation M-value or beta_value
        for each sample in the analysis. The Series name corresponds
        to the probe ID, and the Series is extracted from the meth_data
        DataFrame through a parallellized loop in diff_meth_pos().
    phenotypes:
        A numpy array of binary phenotypes with one phenotype per
        sample (so it must be the same length as probe_data). This is
        the same object as the pheno_data input to diff_meth_pos() after
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
    -995. These rows are processed and removed in the next step of diff_meth_pos() to
    prevent them from interfering with the final analysis and p-value correction
    while printing a list of the unsuccessful probes to alert the user to the issues.
    """
    import warnings
    np.seterr(divide='ignore', over='ignore') # log10(0.0) happens
    warnings.filterwarnings("ignore") # exp(x) overflow error approximates to x=0

    ##Find the probe name for the single pandas series of data contained in probe_data
    probe_ID = probe_data.name
    #groupA = probe_data.loc[ phenotypes['group'] == 0 ]
    #groupB = probe_data.loc[ phenotypes['group'] == 1 ]
    #fold_change = (groupB.mean() - groupA.mean())/groupA.mean()

    try: # fold_change: log2 of ratio of group means
        # M-values are ALREADY log2 transformed, so just use the straight up difference (effect size)
        non_neg = np.array(list((val - probe_data.min())/(probe_data.max()-probe_data.min()) for val in probe_data))
        fold_change = np.log2(non_neg[ phenotypes == 1 ].mean() / non_neg[ phenotypes == 0 ].mean())
        # M-values are ALREADY log2 transformed, so just use the straight up difference (effect size)
        #fold_change = probe_data[ phenotypes == 1 ].mean() / probe_data[ phenotypes == 0 ].mean()
    except ZeroDivisionError as e:
        fold_change = np.log2(non_neg[ phenotypes == 1 ].mean() / 0.001)

    ### ADD INTERCEPT AFTER probes ###
    probe_data = pd.DataFrame(data={'m_value':probe_data})
    probe_data['const'] = 1
    ### add covariates into the probe_data AFTER constant ###
    if isinstance(covariate_data, type(None)):
        pass
    elif isinstance(covariate_data, pd.DataFrame):
        # first, encode strings as ints for Logit.
        for col in covariate_data.columns:
            if covariate_data[col].dtype == 'O':
                covariate_data[col] = covariate_data[col].astype('category').cat.codes
                # normalize codes
                covariate_data[col] = covariate_data[col].apply( lambda x: x/len(covariate_data[col].unique()) )
            else:
                try:
                    # normalize, or, if every value is the same, like a constant with zero range, just set to 1.0
                    cmin = covariate_data[col].min()
                    cmax = covariate_data[col].max()
                    if cmax > cmin:
                        covariate_data[col] = (covariate_data[col] - cmin)/(cmax-cmin)
                    else:
                        covariate_data[col] = 1.0 # blanking a zero-variance column
                except Exception as e:
                    raise Exception(f"Encountered a covariate that could not be normalized: {e}\n{covariate_data[col]}")
            probe_data[col] = list(covariate_data[col]) # merging with index leads to NaNs, but ORDER works.
    elif not isinstance(covariate_data, pd.DataFrame):
        raise Exception(f"covariate data is not a DataFrame")

    ## Fit the logistic model to the individual probe
    # logit = sm.Logit(probe_data, phenotypes, missing='drop')
    #logit_model = sm.GLM(phenotypes, sm.add_constant(probe_data), family=sm.families.Binomial())
    logit_model = sm.GLM(phenotypes, probe_data, family=sm.families.Binomial())
    ## Extract desired statistical measures from logistic fit object
    try:
        #results = logit.fit(disp=debug, warn_convergence=False, method='bfgs') # so if debug is True, display is True
        results = logit_model.fit()
        probe_coef = results.params
        probe_CI = results.conf_int(0.05)  ##returns the lower and upper bounds for the coefficient's 95% confidence interval
        probe_CI = np.array(probe_CI)  ##conf_int returns a pandas dataframe, easier to work with array for extracting results though
        probe_pval = results.pvalues
        probe_SE = results.bse
        ##Fill in the corresponding row of the results dataframe with these values
        probe_stats_row = pd.Series({
            "fold_change": fold_change,
            "Coefficient": -probe_coef[0], # the R GLM logistic method matches this exactly, but the sign is reversed here. dunno why.
            "StandardError": probe_SE[0],
            "PValue": probe_pval[0],
            "95%CI_lower": probe_CI[0][0],
            "95%CI_upper": probe_CI[0][1]},
        name=probe_ID)
    except Exception as e:
        ##If there's a perfect separation error that prevents the model from being fit (like due to small sample sizes),
            ##add that probe name to a list to alert the user later that these probes could not be fit with a logistic regression
        if type(e).__name__ == "PerfectSeparationError":
            probe_stats_row = pd.Series({ #"fold_change": -999,
            "Coefficient":-999,"StandardError":-999,"PValue":-999,"95%CI_lower":-999,"95%CI_upper":-999}, name=probe_ID)
        elif type(e).__name__ == "LinAlgError":
            probe_stats_row = pd.Series({ #"fold_change": -995,
            "Coefficient":-995,"StandardError":-995,"PValue":-995,"95%CI_lower":-995,"95%CI_upper":-995}, name=probe_ID)
        else:
            import traceback;traceback.format_exc()
            raise e
    return probe_stats_row


'''
def scratch_logit(probe_series, pheno, verbose=False, train_fraction=0.9): # pragma: no cover
    """ from https://github.com/PedroDidier/Logistic_Regression/blob/master/DiabetesLogistic_Regression/Logistic_Regression_Diabetes.py
    pass in a probe_series (samples are in rows) and a pheno (list/array with 0 or 1 for group A or B)

    TESTED (2022-03-01) and not reliable. Use the other methods instead.
    Probably the method for optimizing the prediction model isn't great. Doesn't to a loss-min function.
    """
    import pandas as pd
    import numpy as np
    import math
    import scipy.stats

    if isinstance(pheno, pd.DataFrame) and 'group' in pheno.columns:
        pheno = pheno['group'] # drop the 'const' column; not used here.

    #applying z-score on the dataframe
    def standardize_series(ser):
        """ input is a series; returns series of z-scores """
        datatype = ser.dtypes
        if (datatype == 'float64') or (datatype == 'int64'):
            std = ser.std()
            mean = ser.mean()
            ser = (ser - mean) / std
        return ser

    #returns the prediction made by the linear function, already chaged to a probability
    def get_probability_true(coefficients, intercept, k):

        linear_function_out = 0
        for i in range(len(coefficients)):
            #if (i == 0):
            linear_function_out += intercept
            #else:
            linear_function_out += k[i] * coefficients[i]

        #this calculation get's our model result and reduces it to an output value between 0 and 1
        #meaning the probability of diabetes
        return 1 / (1 + (np.power(math.e, -(linear_function_out))))

    #this is where the regression line is figured out
    def calc_coefficients(coefficients, df):
        """ df is a dataframe with 'z' scores and 'p' 0|1 group labels. """
        outcome_mean = df['p'].mean()
        score_mean = df['z'].mean()
        divisor = 0

        # calculating the coefficients for z, leaving the first element on the list for the
        # constant term (set to zero at start)
        for row in range(len(df.index)): # row == one sample probe value, as a z-score
            coefficients += (df['z'][row] - score_mean) * (df['p'][row] - outcome_mean)
            divisor += np.power(df['z'][row] - score_mean, 2)
        coefficients /= divisor # A /= B is equiv to A = A/B ### coefficients is np.array so the one value gets updated inplace each iteration

        # now we calculate the independent/constant term
        #coefficients[0] = outcome_mean
        #coefficients[0] -=  coefficients[1] * score_mean #(AKA df['z'].mean())
        intercept = outcome_mean - coefficients * score_mean
        return coefficients, intercept

    #returns the predicted outcome based on a given probability
    def get_predicted_outcome(func_out):
        # in the test case, using >54% was more useful than 50%, explained by the dataframe's unbalance
        # but 0.54 changed to 0.50 for the general case
        if(func_out > 0.50):
            return 1
        else:
            return 0
    # upstream PREPARING DATA steps
    # sorting values by outcome
    # drop missing; impute
    # df = df.dropna(axis = 0, how = 'any').reset_index(drop = True)

    probe_ID = probe_series.name
    fold_change = ( probe_series[ pheno == 1 ].mean() - probe_series[ pheno == 0 ].mean() ) / probe_series[ pheno == 0 ].mean()
    std_error = probe_series.sem()
    (ci_lower, ci_upper) = DescrStatsW(probe_series).tconfint_mean() # from statsmodels.stats.api

    temp = pd.DataFrame(data={'m': probe_series, 'p': pheno})
    delta_m = round(abs(temp[temp.p == 1]['m'].mean() - temp[temp.p == 0]['m'].mean()),4)
    del temp

    # convert to z-scores
    probe_series = standardize_series(probe_series)

    # add in prediction column
    df = pd.DataFrame(data={'z': probe_series.values, 'p': pheno})

    #remove outliers that would prejudice our fit hyperplane (Z scores must be between -2.5 and +2.5)
    # df = df.loc[(df['z'] < 2.5) & (df['z'] > -2.5)].reset_index(drop = True)

    #dividing the dataset on training and test group; where default train fraction is 80%
    series_N = len(df)
    if series_N < 3:
        raise ValueError("Cannot do logit with less than 3 examples")
    train_N = int(train_fraction * series_N)
    test_N = series_N - train_N
    #test_df = df[:train_N].reset_index(drop = True)
    #train_df = df[train_N:].reset_index(drop = True)
    test_df = df.iloc[train_N:].reset_index(drop = True)
    train_df = df.iloc[:train_N].reset_index(drop = True)
    # print(f"total: {series_N} train: {train_df.shape} test: {test_df.shape}")

    #starting coefficients with 0 and then getting the fit hyperplane ones
    coefficients= [0]
    coefficients, intercept = calc_coefficients(coefficients, train_df)

    #just veryfing results now
    correct = 0
    wrong = 0
    falseneg = 0
    falsepos = 0
    trueneg = 0
    truepos = 0

    avg_prob = []
    for i in range(len(test_df.index)):
        prob = get_probability_true(coefficients, intercept, test_df.loc[i])
        pred_outcome = get_predicted_outcome(prob)
        real_outcome = test_df['p'][i]

        if(pred_outcome == real_outcome):
            if(real_outcome == 1):
                truepos += 1
            else:
                trueneg += 1
            correct += 1
        else:
            if (real_outcome == 0):
                falsepos += 1
            else:
                falseneg += 1
            wrong += 1
        avg_prob.append(prob)
    avg_prob = pd.Series(avg_prob).mean()
    try:
        precision = round((truepos/(truepos + falsepos)),2)
    except ZeroDivisionError:
        precision = -1
    try:
        recall = round((truepos/(truepos + falseneg)),2)
    except ZeroDivisionError:
        recall = -1

    if verbose:
        print(f"Total: {str(correct+wrong)} Correct: {str(correct)} Wrong: {str(wrong)}")
        print(f"True Pos: {str(truepos)} True Neg: {str(trueneg)} False Pos: {str(falsepos)} False Neg: {str(falseneg)}")
        print(f"Accuracy: {str(round(100*(correct/(correct+wrong))))}%")
        print(f"Precision: {str(round(100*(truepos/(truepos + falsepos))))}%")
        print(f"Recall: {str(round(100*(truepos/(truepos + falseneg))))}%")

    return pd.Series({
        "fold_change": math.log2(((1/avg_prob) - 1)), # actually, this is the log2(odds), but seems more useful
        "Coefficient": coefficients[0],
        "StandardError": std_error,
        "PValue": scipy.stats.norm.sf(abs(probe_series)).mean()*2, # *2 for two-tailed, then mean() because each z-score per sample is returned. #-- https://www.statology.org/p-value-from-z-score-python/
        "95%CI_lower": ci_lower,
        "95%CI_upper": ci_upper,
        "intercept": intercept[0],
        "accuracy": round((correct/(correct+wrong)),2),
        "precision": precision,
        "recall": recall,
        "delta_m": delta_m # the difference between group(0) avg and group(1) avg M-value.
        }, name=probe_ID)
'''

##########################################
##########################################
##########################################

def volcano_plot(stats_results, **kwargs):
    """
This function writes the pandas DataFrame output of diff_meth_pos() to a CSV file
named by the user. The DataFrame has a row for every successfully tested probe
and columns with different regression statistics as follows:
        - regression coefficient
        - lower limit of the coefficient's 95% confidence interval
        - upper limit of the coefficient's 95% confidence interval
        - standard error
        - p-value
        - FDR q-value (p-values corrected for multiple testing using the Benjamini-Hochberg FDR method)

Inputs and Parameters:

    stats_results (required):
        A pandas DataFrame output by the function diff_meth_pos().
    'alpha':
        Default: 0.05, The significance level that will be used to highlight the most
        significant p-values (or adjusted FDR Q-values) on the plot.
    'cutoff':
        the beta-coefficient cutoff | Default: None
        format: a list or tuple with two numbers for (min, max) or 'auto'.
        If specified in kwargs, will exclude values within this range of regression coefficients OR fold-change range
        from being "significant" and put dotted vertical lines on chart.
        'auto' will select a beta coefficient range that excludes 95% of results from appearing significant.
    'adjust':
        (default False) -- if this will adjust the p-value cutoff line for false discovery rate (Benjamini-Hochberg).
        Use 'fwer' to set the target rate.
    'fwer':
        family-wise error rate (default is 0.1) -- specify a probability [0 to 1.0] for false discovery rate
    'data_type_label':
        What to put on X-axis. Either 'Fold Change' (default) or 'Regression Coefficient'.
    visualization kwargs:
        - `palette` -- color pattern for plot -- default is [blue, red, grey]
            other palettes: ['default', 'Gray', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c', 'Gray2', 'Gray3']
        - `width` -- figure width -- default is 16
        - `height` -- figure height -- default is 8
        - `fontsize` -- figure font size -- default 16
        - `dotsize` -- figure dot size on chart -- default 30
        - `border` -- plot border --  default is OFF
        - `data_type_label` -- (e.g. Beta Values, M Values) -- default is 'Beta'
        - `plot_cutoff_label` -- add label to dotted line on plot -- default False
    save:
        specify that it export an image in `png` format.
        By default, the function only displays a plot.
    filename:
        specify an export filename. default is `volcano_<current_date>.png`.

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
    verbose = False if kwargs.get('verbose') == False else True # if ommited, verbose is default ON
    if kwargs.get('palette') in color_schemes:
        colors = color_schemes[kwargs.get('palette')]
    else:
        colors = color_schemes['Volcano']
    colors = list(colors.colors)
    if kwargs.get('palette') and kwargs.get('palette') not in color_schemes:
        print(f"WARNING: user supplied color palette {kwargs.get('palette')} is not a valid option! (Try: {list(color_schemes.keys())})")
    alpha = 0.05 if not kwargs.get('alpha') else kwargs.get('alpha')
    bcutoff = kwargs.get('cutoff', None)
    if bcutoff == 'auto':
        #bcutoff = (0.95*stats_results['Coefficient'].min(), 0.95*stats_results['Coefficient'].max())
        # biostars: 2-fold change is a reasonable cutoff
        bcutoff = (-1,1)

    def_width = int(kwargs.get('width',16))
    def_height = int(kwargs.get('height',8))
    def_fontsize = int(kwargs.get('fontsize',16))
    def_dot_size = int(kwargs.get('dotsize',30))
    border = True if kwargs.get('border') == True else False # default OFF
    if kwargs.get('data_type_label'):
        data_type_label = kwargs.get('data_type_label')
    elif 'fold_change' in stats_results.columns:
        data_type_label = 'Fold Change' # --- FIX --- '$log_{2} Fold Change$'
    else:
        data_type_label = 'Regression Coefficient'
    save = True if kwargs.get('save') else False
    plot_cutoff_label = kwargs.get('plot_cutoff_label', False)
    adjust = kwargs.get('adjust', False)

    if bcutoff != None and type(bcutoff) in (list,tuple) and len(bcutoff) == 2:
        pre = len(stats_results)
        if data_type_label == 'Fold Change':
            retained_stats_results = stats_results[
                (stats_results['fold_change'] < bcutoff[0])
                | (stats_results['fold_change'] > bcutoff[1])
                ].index
        elif data_type_label == 'Regression Coefficient':
            retained_stats_results = stats_results[
                (stats_results['Coefficient'] < bcutoff[0])
                | (stats_results['Coefficient'] > bcutoff[1])
                ].index
        else:
            raise ValueError(f"data_type_label must be one of (Fold Change, Regression Coefficient)")
        print(f"Excluded {pre-len(retained_stats_results)} probes outside of the specified beta coefficient range: {bcutoff}")
    elif bcutoff != None:
        print(f'WARNING: Your beta_coefficient_cutoff value ({bcutoff}) is invalid. Pass a list or tuple with two values for (min,max).')
        bcutoff = None # prevent errors below

    if adjust: # FDR adjustment
        prev_pvalue_cutoff_y = -np.log10(alpha)
        cutoff_adjusted = sm.stats.multipletests(probe_stats["PValue"], alpha=kwargs.get('fwer',0.1), method="fdr_bh")
        pvalue_cutoff_y = -np.log10(cutoff_adjusted[3])
        total_sig_probes = sum(cutoff_adjusted[0])
        if verbose:
            print(f"p-value cutoff adjusted: {bcutoff} ({prev_pvalue_cutoff_y}) ==[ fdr_bh ]==> {cutoff_adjusted[3]} ({pvalue_cutoff_y}) | {total_sig_probes} probes significant")
    else:
        pvalue_cutoff_y = -np.log10(alpha)


    change_col = 'fold_change' if 'fold_change' in stats_results else 'Coefficient'
    # statistic_col = 'FDR_QValue' if adjust is True else 'PValue'
    statistic_col = 'PValue' # we adjust the significant cutoff line; we don't change the column of data shown
    # colors are 0=red, 1=blue, 2=silver
    palette = []
    for i in range(len(stats_results)):
        if -np.log10(stats_results[statistic_col][i]) > pvalue_cutoff_y: # ">" because already -log10 transformed
            if stats_results[change_col][i] > 0:
                if bcutoff and stats_results[change_col][i] > bcutoff[1]: # red if coef > positive-x-cutoff
                    palette.append(colors[0])
                elif bcutoff:
                    palette.append(colors[2])
                else:
                    palette.append(colors[0]) # red if no beta-coef filtering applied
            else:
                if bcutoff and stats_results[change_col][i] < bcutoff[0]: # blue if coef < negative-x-cutoff
                    palette.append(colors[1])
                elif bcutoff:
                    palette.append(colors[2])
                else:
                    palette.append(colors[1]) # blue if no beta-coef filtering applied
        else:
            palette.append(colors[2])
    plt.rcParams.update({'font.family':'sans-serif', 'font.size': def_fontsize})
    fig = plt.figure(figsize=(def_width,def_height))
    ax = fig.add_subplot(111)
    #print(f"DEBUG fold-change OR beta-coef range: {stats_results[change_col].min()} {stats_results[change_col].max()}")
    #print(f"DEBUG {statistic_col} range: {stats_results[statistic_col].min()} {stats_results[statistic_col].max()}")
    fraction_most_significant = len(stats_results[ stats_results[statistic_col] == stats_results[statistic_col].min() ])/len(stats_results)
    #print(f"DEBUG {round(100*fraction_most_significant)}% of {len(stats_results)} probes have the lowest {statistic_col}: {stats_results[statistic_col].min()}")
    plt.scatter(stats_results[change_col], # fold_change
        -np.log10(stats_results[statistic_col]),
        c=palette,
        s=def_dot_size)
    fig.get_axes()[0].set_xlabel(data_type_label)
    if statistic_col == 'PValue':
        fig.get_axes()[0].set_ylabel("$-log_{10}$( p-value )")
    else:
        fig.get_axes()[0].set_ylabel("$-log_{10}$( FDR Adjusted Q Value )")
    ax.axhline(y=pvalue_cutoff_y, color="grey", linestyle='--')
    if plot_cutoff_label:
        plt.text(stats_results[change_col].min(), pvalue_cutoff_y, f'p-value: {round(pvalue_cutoff_y, 2)}', color="grey")

    if bcutoff and type(bcutoff) in (list,tuple):
        ax.axvline(x=bcutoff[0], color="grey", linestyle='--')
        ax.axvline(x=bcutoff[1], color="grey", linestyle='--')
    # hide the border; unnecessary
    if border == False:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    has_sig_probes = False if len(stats_results[ stats_results[statistic_col] <= kwargs.get('fwer',0.1) ]) > 0 else True
    if has_sig_probes: # label these, up to 100 of them
        top_probes = stats_results.sort_values(['FDR_QValue','PValue'], ascending=(True,True)).head(30)
        text_labels = []
        #counted = 1
        top_probes['ind'] = range(len(top_probes))
        top_probes['minuslog10value'] = -np.log10(top_probes[statistic_col])
        for pname,probe in top_probes.iterrows():
            if probe[statistic_col] < kwargs.get('fwer',0.1): # or counted <= 10:
                text_labels.append( plt.text(probe.ind, probe.minuslog10value, pname, fontsize='x-small', fontweight='light') )
            #counted += 1
        adjust_text(text_labels)

    if save:
        filename = kwargs.get('filename') if kwargs.get('filename') else f"volcano_{len(stats_results)}_{str(datetime.date.today())}.png"
        plt.savefig(filename)
        if verbose == True:
            print(f"saved {filename}")
    if verbose == True:
        plt.show()
    else:
        plt.close(fig)


def manhattan_plot(stats_results, array_type, **kwargs):
    """
In EWAS Manhattan plots, epigenomic probe locations are displayed along the X-axis,
with the negative logarithm of the association P-value for each single nucleotide polymorphism
(SNP) displayed on the Y-axis, meaning that each dot on the Manhattan plot signifies a SNP.
Because the strongest associations have the smallest P-values (e.g., 10−15),
their negative logarithms will be the greatest (e.g., 15).

GWAS vs EWAS
============
    - genomic coordinates along chromosomes vs epigenetic probe locations along chromosomes
    - p-values are for the probe value associations, using linear or logistic regression,
    between phenotype A and B.

Ref
===
    Hints of hidden heritability in GWAS. Nature 2010. (https://www.ncbi.nlm.nih.gov/pubmed/20581876)

Required Inputs
===============
    stats_results:
        a pandas DataFrame containing the stats_results from the linear/logistic regression run on m_values or beta_values
        and a pair of sample phenotypes. The DataFrame must contain A "PValue" column. the default output of diff_meth_pos() will work.
    array_type:
        specify the type of array [450k, epic, epic+, mouse, 27k], so that probes can be mapped to chromosomes.

output kwargs
=============
    save:
        specify that it export an image in `png` format.
        By default, the function only displays a plot.
    filename:
        specify an export filename. The default is `f"manhattan_<stats>_<timestamp>.png"`.


visualization kwargs
====================
    - `verbose` (True/False) - default is True, verbose messages, if omitted.
    - `fwer` (default 0.1) familywise error rate (fwer) is used to set p-value threshold and the FDR threshold line.
    - `genome_build` -- NEW or OLD. Default is NEWest genome_build.
    - `label-prefix` -- how to refer to chromosomes. By default, it shows numbers like 1 ... 22, and X, Y.
        pass in 'CHR-' to add a prefix to plot labels, or rename with 'c' like: c01 ... c22.

    There are some preset "override" options for threshold lines on plots:

    - `explore`: (default False) include all FOUR significance threshold lines on plot
        (suggestive, significant, bonferroni, and false discovery rate). Useful for data exploration.
    - `no_thresholds`: (default False) set to True to hide all FOUR threshold lines from plot.
    - `plain`: (default False) hide all lines AND don't label significant probes.
    - `statsmode`: (default False) show the FDR and Bonferroni thresholds, hide the suggestive and genomic significant lines.

    These allow you to toggle lines/labels on or off:

    - `fdr`: (default False) draw a threshold line on plot corresponding to the adjusted false discovery rate = 0.05
    - `bonferroni`: (default False) draw the Bonferroni threshold, correcting for multiple comparisons.
    - `suggestive`: (default 1e-5) draw the consensus "suggestive significance" theshold at p < 1e-5,
        or set to False to hide.
    - `significant`: (default 5e-8) draw the consensus "genomic significance" theshold at p < 5e-8,
        or set to False to hide.
    - `plot_cutoff_label` (default True) label to each dotted line on the plot
    - `label_sig_probes` (default True) labels significant probes showing the greatest difference between groups.

    Chart options:

    - `ymax` -- default: 50. Set to avoid plotting extremely high -10log(p) values.
    - `width` -- figure width -- default is 16
    - `height` -- figure height -- default is 8
    - `fontsize` -- figure font size -- default 16
    - `border` -- plot border --  default is OFF
    - `palette` -- specify one of a dozen options for colors of chromosome regions on plot:
      ['default', 'Gray', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3',
      'tab10', 'tab20', 'tab20b', 'tab20c', 'Gray2', 'Gray3']
    """
    allowed = ['verbose','width','height','fontsize','fdr','border','save','palette',
    'fwer','ymax','plot_cutoff_label','label_sig_probes','suggestive','significant','fdr',
    'bonferroni','explore','no_thresholds','plain','statsmode','genome_build','label_prefix',
    'filename','save']
    if any([k for k in kwargs if k not in allowed]):
        misspelled = ', '.join([k for k in kwargs if k not in allowed])
        raise KeyError(f"Unrecognized argument(s): {misspelled}")

    verbose = False if kwargs.get('verbose') == False else True
    def_width = int(kwargs.get('width',16))
    def_height = int(kwargs.get('height',8))
    def_fontsize = int(kwargs.get('fontsize',12))
    border = True if kwargs.get('border') == True else False
    save = True if kwargs.get('save') else False
    fwer = float(kwargs.get('fwer', 0.1))
    pvalue_cutoff_y = -np.log10(fwer)
    ymax = kwargs.get('ymax',50)
    plot_cutoff_label = kwargs.get('plot_cutoff_label',True)
    label_sig_probes = kwargs.get('label_sig_probes',True)
    suggestive = kwargs.get('suggestive', 1e-5) # literature also uses 5e-7 here
    significant = kwargs.get('significant', 5e-8)
    fdr = kwargs.get('fdr', False)
    bonferroni = kwargs.get('bonferroni', False)
    if kwargs.get('explore') == True:
        suggestive = kwargs.get('suggestive', 1e-5) # if explore is True and suggestive is False, it won't display.
        significant = kwargs.get('significant', 5e-8)
        fdr = kwargs.get('fdr', True)
        bonferroni = kwargs.get('bonferroni', True)
    if kwargs.get('no_thresholds') == True:
        suggestive, significant, fdr, bonferroni = (False, False, False, False)
    if kwargs.get('plain') == True:
        suggestive, significant, fdr, bonferroni, label_sig_probes = (False, False, False, False, False)
    if kwargs.get('statsmode') == True:
        suggestive, significant, fdr, bonferroni = (False, False, True, True)
    if kwargs.get('palette'):
        if kwargs.get('palette') not in color_schemes:
            print(f"WARNING: user supplied color palette {kwargs.get('palette')} is not a valid option! (Try: {list(color_schemes.keys())})")
            colors = list(color_schemes['default'].colors)
        else:
            colors = list(color_schemes[kwargs.get('palette')].colors)
    else:
        colors = list(color_schemes['Gray'].colors)

    df = stats_results

    if 'PValue' not in df.columns:
        raise KeyError(f"stats dataframe must have a `PValue` column.")
    NL = -np.log10(df.PValue)
    NL[NL == np.inf] = -1
    NL[NL == -1] = min(np.argmax(NL),ymax) # replacing inf; capping at ymax
    df['minuslog10value'] = NL

    #### MAP probes to chromosomes using NEW or OLD genome build ####
    pre_length = len(df)
    array_types = {'450k', 'epic', 'mouse', '27k', 'epic+'}
    if isinstance(array_type, methylprep.Manifest):
        manifest = array_type # faster to pass manifest in, if doing a lot of plots
        array_type = str(manifest.array_type)
    elif array_type.lower() not in array_types:
            raise ValueError(f"Specify your array_type as one of {array_types}; '{array_type.lower()}' was not recognized.")
    else:
        manifest = methylprep.Manifest(methylprep.ArrayType(array_type), verbose=False)
    mapinfo_col = 'OLD_MAPINFO' if kwargs.get('genome_build') == 'OLD' else 'MAPINFO'
    probe2chr = create_probe_chr_map(manifest, genome_build=kwargs.get('genome_build',None))
    mapinfo_df = create_mapinfo(manifest, genome_build=kwargs.get('genome_build',None))

    if kwargs.get('label_prefix') == None:
        # values are CHR-01, CHR-02, .. CHR-22, CHR-X... make 01, 02, .. 22 by default.
        df['chromosome'] = df.index.map(lambda x: probe2chr.get(x).replace('CHR-','') if probe2chr.get(x) else None)
    elif kwargs.get('label_prefix') != None:
        prefix = kwargs.get('label_prefix')
        df['chromosome'] = df.index.map(lambda x: probe2chr.get(x).replace('CHR-',prefix) if probe2chr.get(x) else None)

    # drop probes not in manifest from plot and warn
    NaNs = 0
    if len(df[df['chromosome'].isna() == True]) > 0:
        NaNs = len(df[df['chromosome'].isna() == True])
        print(f"{NaNs} NaNs dropped")
        df.dropna(subset=['chromosome'], inplace=True)
    if (len(df) + NaNs) < pre_length and verbose:
        print(f"Warning: {pre_length - len(df)} probes were removed because their names don't match methylize's lookup list")
    # drop any manifest probes that aren't in stats
    df['MAPINFO']= mapinfo_df.loc[mapinfo_df.index.isin(df.index)][[mapinfo_col]]
    df = df.sort_values('MAPINFO')
    df = df.sort_values('chromosome')

    # How to plot gene vs. -log10(pvalue) and colour it by chromosome?
    df['ind'] = range(len(df)) # adds an index column, separate from probe names
    df_grouped = df.groupby(('chromosome'))
    # print('Total probes to plot:', len(df['ind']))
    plt.rcParams.update({'font.family':'sans-serif', 'font.size': def_fontsize})
    fig = plt.figure(figsize=(def_width,def_height))
    ax = fig.add_subplot(111)
    x_labels = []
    x_labels_pos = []
    # print(" | ".join([f"{name} {len(group)}" for name,group in df_grouped]))
    for num, (name, group) in enumerate(df_grouped):
        try:
            repeat_color = colors[num % len(colors)]
            group.plot(kind='scatter', x='ind', y='minuslog10value', color=repeat_color, ax=ax)
            x_labels.append(name)
            x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2))
        except ValueError as e:
            print(e)

    def add_cutoff_line(df, ax, arbitrary_value=None, color='grey', label=None):
        margin_padding = min([10 + int(len(df.index)/5000.0), 50])
        if arbitrary_value:
            cutoff = -np.log10(arbitrary_value)
        else:
            raise ValueError("You must provide a cutoff p-value (it will be -log10 transformed)")
        xy_line = {'x':list(range(len(df))), 'y': [cutoff for i in range(len(df))]}
        pd.DataFrame(xy_line).plot(kind='line', x='x', y='y', color=color, ax=ax, legend=False, style='--')
        if label:
            plt.text(margin_padding, cutoff + (0.01 * cutoff), label, color=color)

    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlim([0, len(df)])

    if not isinstance(suggestive, bool) and suggestive != False:
        add_cutoff_line(df, ax, arbitrary_value=suggestive, color='red', label=suggestive)
    if not isinstance(significant, bool) and significant != False:
        add_cutoff_line(df, ax, arbitrary_value=significant, color='blue', label=significant)
    if bonferroni:
        bonferroni_cutoff = sm.stats.multipletests(df["PValue"], alpha=fwer, method='bonferroni')[3]
        bonferroni_cutoff_label = "{:.2e}".format(bonferroni_cutoff)
        add_cutoff_line(df, ax, arbitrary_value=bonferroni_cutoff, color='gray', label=f"Bonferroni: {bonferroni_cutoff_label}")
    # find the p-value where FDR-Q ~ 0.05
    no_fdr_probes = None
    try:
        # --v1.0 wrong version-- fdr_cutoff = df[["PValue","FDR_QValue"]][df.PValue <= 0.05].sort_values("FDR_QValue", ascending=True)
        # FDR cutoff line is the p-value corresponding to FDR = 0.05.
        # "Find the largest (unadjusted) p-value for which the FDR is below the desired level. Draw the line at that value of p."
        fdr_cutoff = df[ df.FDR_QValue <= fwer ].sort_values('PValue', ascending=False).PValue.max()
        if fdr_cutoff is np.nan:
            no_fdr_probes = True
            raise Exception("No significant probes")
        fdr_label = "{:.2e}".format(fdr_cutoff)
        if fdr:
            add_cutoff_line(df, ax, arbitrary_value=fdr_cutoff, color='black', label=f"FDR: {fdr_label}")
        no_fdr_probes = False
    except Exception as e:
        print(f"Error: {e} (FDR line omitted from plot)")

    if no_fdr_probes:
        pass
    elif label_sig_probes:
        # label top 10 probes, or if q < 0.01; need (x,y on existing plot: x=ind, y=minuslog10value)
        top_probes = df.sort_values(['FDR_QValue','PValue'], ascending=(True,True)).head(30)
        text_labels = []
        #counted = 1 #--- used if you want to show ANY of the best probes, regardless of statistics.
        for pname,probe in top_probes.iterrows():
            #plt.annotate(probe.MAPINFO, (probe.ind, probe.minuslog10value), xytext=(0,30), textcoords='offset points',
            #    arrowprops={'arrowstyle':'-', 'color':'black'}) #{'width':1, 'frac':1, 'headwidth':1, 'shrink':0.05})
            if probe.FDR_QValue <= fwer: # or counted <= 10:
                text_labels.append( plt.text(probe.ind, probe.minuslog10value, pname) )
            #counted += 1
        adjust_text(text_labels, only_move={'points':'y', 'text':'y'})

    # adjust max height to ensure dotted cutoff line appears
    highest_value = max([max(df['minuslog10value']), -np.log10(5e-8)])
    if pvalue_cutoff_y > ymax and verbose:
        LOGGER.warning(f"Adjusted significance line is above ymax, and won't appear.")
    ax.set_ylim([0, highest_value + 0.05 * highest_value])
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('$-log_{10}$(p)')
    # hide the border; unnecessary
    if border == False:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    if kwargs.get('label_prefix') != None:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    if save:
        filename = kwargs.get('filename') if kwargs.get('filename') else f"manhattan_{len(stats_results)}_{str(datetime.date.today())}.png"
        plt.savefig(filename)
        if verbose == True:
            print(f"saved {filename}")
    if verbose == True:
        plt.show()
    else:
        plt.close(fig)


def probe_corr_plot(stats, group='sig', colorby='pval'): # pragma: no cover
    """
    - group='sig' is default (using PValue < 0.05)
    - group='chromosome' also kinda works.
    - colorby= pval or FDR; what to use to color the significant probes, if group='sig'
    """
    import matplotlib.pyplot as plt
    temp = stats.sort_values('Coefficient') # puts uncorrelated probes in the middle
    temp['x'] = range(len(temp.index))
    fig,ax = plt.subplots(1,1, figsize=(12,8))
    if colorby == 'pval':
        temp['sig'] = temp.apply(lambda row: row['PValue'] < 0.05, axis=1)
    elif colorby == 'FDR':
        temp['sig'] = temp.apply(lambda row: row['FDR_QValue'] < 0.05, axis=1)
    groups = temp.groupby(group)
    if group == 'sig':
        colors = {True:'tab:green', False:'tab:blue'}
        for (label, _group) in groups:
            ax.scatter(_group.x, _group.Coefficient, color=colors[label], s=3)
    elif group == 'chromosome':
        colors = color_schemes['default']
        colors = list(colors.colors)
        index = 0
        for num, (label, _group) in enumerate(groups):
            ind_sub = [index + i for i in range(len(_group))]
            repeat_color = colors[num % len(colors)]
            ax.scatter(ind_sub, _group['Coefficient'], color=repeat_color, s=3)
            index += len(ind_sub)
    ax.fill_between(temp['x'], temp['95%CI_lower'], temp['95%CI_upper'], color='gray', alpha=0.2)
    ax.set_xlabel('probe index')
    ax.set_ylabel('probe correlation between groups (r)')
    plt.show()


################################################

"""


def manhattan_plot_deprecated(stats_results, array_type, **kwargs):
In EWAS Manhattan plots, epigenomic probe locations are displayed along the X-axis,
with the negative logarithm of the association P-value for each single nucleotide polymorphism
(SNP) displayed on the Y-axis, meaning that each dot on the Manhattan plot signifies a SNP.
Because the strongest associations have the smallest P-values (e.g., 10−15),
their negative logarithms will be the greatest (e.g., 15).

GWAS vs EWAS
============
    - genomic coordinates along chromosomes vs epigenetic probe locations along chromosomes
    - p-values are for the probe value associations, using linear or logistic regression,
    between phenotype A and B.

Ref
===
    Hints of hidden heritability in GWAS. Nature 2010. (https://www.ncbi.nlm.nih.gov/pubmed/20581876)

Required Inputs
===============
    stats_results:
        a pandas DataFrame containing the stats_results from the linear/logistic regression run on m_values or beta_values
        and a pair of sample phenotypes. The DataFrame must contain A "PValue" column. the default output of diff_meth_pos() will work.
    array_type:
        specify the type of array [450k, epic, epic+, mouse, 27k], so that probes can be mapped to chromosomes.

output kwargs
=============
    save:
        specify that it export an image in `png` format.
        By default, the function only displays a plot.
    filename:
        specify an export filename. The default is `f"manhattan_<stats>_<timestamp>.png"`.


visualization kwargs
====================

    - `verbose` (True/False) - default is True, verbose messages, if omitted.
    - `genome_build` -- NEW or OLD. Default is NEWest genome_build.
    - `width` -- figure width -- default is 16
    - `height` -- figure height -- default is 8
    - `fontsize` -- figure font size -- default 16
    - `border` -- plot border --  default is OFF
    - `palette` -- specify one of a dozen options for colors of chromosome regions on plot:
      ['default', 'Gray', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3',
      'tab10', 'tab20', 'tab20b', 'tab20c', 'Gray2', 'Gray3']
    - `cutoff` -- threshold p-value for where to draw a line on the plot (default: 5x10^-8 on plot, or p<=0.05)
        specify a number, such as 0.05.
    - `label-prefix` -- how to refer to chromosomes. By default, it shows numbers like 1 ... 22, and X, Y.
        pass in 'CHR-' to add a prefix to plot labels, or rename with 'c' like: c01 ... c22.
    - `adjust`: (True, False, string)
        By default, the cutoff line is adjusted for multiple tests using Yoav Benjamini and Yosef Hochberg False Discovery Rate (FDR).
        This correction is applied after regression to control for alpha. Setting to True moves the
        dotted significance line on the plot upward -- to a more conservative threshold than 0.05 -- to account
        for multiple comparisons. Multiple comparisons increase the chance of "seeing" a significant difference when one does not truly
        exist, and DMP runs tens-of-thousands of comparisons across all probes. To disable, set `adjust=None`
        and the dotted line will remain at 0.05 and NOT control for multiple tests. Or, if you set to a string,
        (any of the correction methods listed in https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html)
        it will use that method instead of `fdr_bh`. These options include:
        - bonferroni : one-step correction
        - sidak : one-step correction
        - holm-sidak : step down method using Sidak adjustments
        - holm : step-down method using Bonferroni adjustments
        - simes-hochberg : step-up method (independent)
        - hommel : closed method based on Simes tests (non-negative)
        - fdr_bh : Benjamini/Hochberg (non-negative)
        - fdr_by : Benjamini/Yekutieli (negative)
        - fdr_tsbh : two stage fdr correction (non-negative)
        - fdr_tsbky : two stage fdr correction (non-negative)
    - `ymax` -- default: 50. Set to avoid plotting extremely high -10log(p) values.
    - `fdr`: plot FDR_QValue instead of PValues on plot.
    - `plot_cutoff_label` -- default True: adds a label to the dotted line on the plot, unless set to False

    verbose = False if kwargs.get('verbose') == False else True # if ommited, verbose is default ON
    def_width = int(kwargs.get('width',16))
    def_height = int(kwargs.get('height',8))
    def_fontsize = int(kwargs.get('fontsize',12))
    fdr = True if kwargs.get('fdr') == True else False
    # def_dot_size = int(kwargs.get('dotsize',16)) -- df.groupby.plots don't accept this.
    border = True if kwargs.get('border') == True else False # default OFF
    save = True if kwargs.get('save') else False
    if kwargs.get('palette') in color_schemes:
        colors = color_schemes[kwargs.get('palette')]
    else:
        colors = color_schemes['Gray']
    if kwargs.get('palette') and kwargs.get('palette') not in color_schemes:
        print(f"WARNING: user supplied color palette {kwargs.get('palette')} is not a valid option! (Try: {list(color_schemes.keys())})")
    if kwargs.get('cutoff'):
        alpha = float(kwargs.get('cutoff'))
    else:
        alpha = 0.05
    ymax = kwargs.get('ymax',50)
    pvalue_cutoff_y = -np.log10(alpha)
    plot_cutoff_label = kwargs.get('plot_cutoff_label',True)
    adjust = kwargs.get('adjust',True)

    df = stats_results

    if 'FDR_QValue' not in df.columns and 'PValue' not in df.columns:
        raise KeyError(f"stats dataframe muste ither have a `FDR_QValue` or `PValue` column.")
    #if kwargs.get('fdr') and 'FDR_QValue' not in df.columns:
    #    LOGGER.warning("FDR specified but no `FDR_QValue` column in stats data. Using PValue instead.")
    # get -log_10(PValue) -- but set any p 0.000 to the highest value found, to avoid NaN/inf
    #if kwargs.get('fdr') and 'FDR_QValue' in df.columns:
    #    NL = -np.log10(df.FDR_QValue)
    #else:
    NL = -np.log10(df.PValue)
    NL[NL == np.inf] = -1
    NL[NL == -1] = min(np.argmax(NL),ymax) # replacing inf; capping at ymax (100)
    df['minuslog10pvalue'] = NL

    # map probes to chromosome using an internal methylize lookup pickle, probe2chr.
    pre_length = len(df)

    array_types = {'450k', 'epic', 'mouse', '27k', 'epic+'}
    if isinstance(array_type, methylprep.Manifest):
        manifest = array_type # faster to pass manifest in, if doing a lot of plots
        array_type = str(manifest.array_type)
    elif array_type.lower() not in array_types:
            raise ValueError(f"Specify your array_type as one of {array_types}; '{array_type.lower()}' was not recognized.")
    else:
        manifest = methylprep.Manifest(methylprep.ArrayType(array_type))

    probe2chr = create_probe_chr_map(manifest, genome_build=kwargs.get('genome_build',None))
    mapinfo_df = create_mapinfo(manifest, genome_build=kwargs.get('genome_build',None))

    if kwargs.get('label_prefix') == None:
        # values are CHR-01, CHR-02, .. CHR-22, CHR-X... make 01, 02, .. 22 by default.
        df['chromosome'] = df.index.map(lambda x: probe2chr.get(x).replace('CHR-','') if probe2chr.get(x) else None)
    elif kwargs.get('label_prefix') != None:
        prefix = kwargs.get('label_prefix')
        df['chromosome'] = df.index.map(lambda x: probe2chr.get(x).replace('CHR-',prefix) if probe2chr.get(x) else None)

    NaNs = 0
    if len(df[df['chromosome'].isna() == True]) > 0:
        NaNs = len(df[df['chromosome'].isna() == True])
        print(f"{NaNs} NaNs dropped")
        df.dropna(subset=['chromosome'], inplace=True)
    # in the case that probes are not in the lookup, this will drop those probes from the chart and warn user.
    if (len(df) + NaNs) < pre_length and verbose:
        print(f"Warning: {pre_length - len(df)} probes were removed because their names don't match methylize's lookup list")

    # BELOW: causes an "x axis needs to be numeric" error.
    #df.chromosome = df.chromosome.astype('category')
    #df.chromosome = df.chromosome.cat.set_categories([i for i in range(0,23)], ordered=True)
    df['MAPINFO']= mapinfo_df.loc[mapinfo_df.index.isin(df.index)][['MAPINFO']] # drop any manifest probes that aren't in stats
    df = df.sort_values('MAPINFO')
    df = df.sort_values('chromosome')

    # How to plot gene vs. -log10(pvalue) and colour it by chromosome?
    df['ind'] = range(len(df)) # adds an index column, separate from probe names
    df_grouped = df.groupby(('chromosome'))
    print('Total probes to plot:', len(df['ind']))
    # make the figure. set defaults first.
    #plt.rc({'family': 'sans-serif', 'size': def_fontsize}) -- this gets overridden by volcano settings in notebook.
    plt.rcParams.update({'font.family':'sans-serif', 'font.size': def_fontsize})
    fig = plt.figure(figsize=(def_width,def_height))
    ax = fig.add_subplot(111)
    colors = list(colors.colors)
    x_labels = []
    x_labels_pos = []
    print(" | ".join([f"{name} {len(group)}" for name,group in df_grouped]))
    for num, (name, group) in enumerate(df_grouped):
        try:
            repeat_color = colors[num % len(colors)]
            group.plot(kind='scatter', x='ind', y='minuslog10pvalue', color=repeat_color, ax=ax)
            x_labels.append(name)
            x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2))
            #print('DEBUG', num, name, group.shape)
        except ValueError as e:
            print(e)

    def add_cutoff_line(df, ax, adjust_method=None, arbitrary_value=None, color='grey', label=None):
        if arbitrary_value:
            cutoff = -np.log10(arbitrary_value)
        elif arbitrary_value == None and adjust_method == None:
            raise ValueError("Either provide a cutoff value or a correction method")
        else:
            adjusted = sm.stats.multipletests(df["PValue"], alpha=alpha, method=adjust_method)[3]
            cutoff = -np.log10(adjusted)
        xy_line = {'x':list(range(len(df))), 'y': [cutoff for i in range(len(df))]}
        pd.DataFrame(xy_line).plot(kind='line', x='x', y='y', color=color, ax=ax, legend=False, style='--')
        if label:
            plt.text(10, cutoff + (0.01 * cutoff), label, color=color)

    if adjust: # True, False, None, str
        if isinstance(adjust, str):
            adjust_method = adjust
        else:
            adjust_method = 'fdr_bh'
        prev_pvalue_cutoff_y = pvalue_cutoff_y
        adjusted = sm.stats.multipletests(probe_stats["PValue"], alpha=alpha, method=adjust_method)[3]
        # multipletests(stats_results.PValue, alpha=alpha)
        pvalue_cutoff_y = -np.log10(adjusted)
        if verbose:
            print(f"p-value cutoff adjusted: {prev_pvalue_cutoff_y} ==[ {adjust_method} ]==> {pvalue_cutoff_y}")
        # draw the p-value cutoff line
        xy_qline = {'x':list(range(len(stats_results))), 'y': [pvalue_cutoff_y for i in range(len(stats_results))]}
        df_qline = pd.DataFrame(xy_qline)

    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlim([0, len(df)])

    add_cutoff_line(df, ax, adjust_method=None, arbitrary_value=5e-8, color='red')
    add_cutoff_line(df, ax, adjust_method='bonferroni', color='pink', label='bonferroni')
    add_cutoff_line(df, ax, adjust_method='fdr_bh', color='blue', label='FDR')
    highest_value = max([max(df['minuslog10pvalue']), -np.log10(5e-8)])

    # adjust max height to ensure dotted cutoff line appears
    # highest_value = max(df['minuslog10pvalue']) if pvalue_cutoff_y < max(df['minuslog10pvalue']) else pvalue_cutoff_y
    # adjust max height to below the absolute ymax
    highest_value = highest_value if highest_value < ymax else ymax
    if pvalue_cutoff_y > ymax and verbose:
        LOGGER.warning(f"{adjust_method} adjusted significance line is above ymax, and won't appear.")
    ax.set_ylim([0, highest_value + 0.05 * highest_value])
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('-log(p)')

    if plot_cutoff_label == True: # False hides both labels and the gray dotted bonferroni line
        plt.text(10, pvalue_cutoff_y + (0.01*pvalue_cutoff_y), f'FDR q=0.05', color="red")
        bonferroni = sm.stats.multipletests(probe_stats["PValue"], alpha=alpha, method='bonferroni')[3]
        blog = -np.log10(bonferroni)
        print(blog, bonferroni, pvalue_cutoff_y)
        xy_bline = {'x':list(range(len(stats_results))), 'y': [blog for i in range(len(stats_results))]}
        df_bline = pd.DataFrame(xy_bline)
        plt.text(10, blog + (0.01 * blog), f'bonferroni', color="gray")
        df_bline.plot(kind='line', x='x', y='y', color='blue', ax=ax, legend=False, style='--')

    # hide the border; unnecessary
    if border == False:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    if kwargs.get('label_prefix') != None:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    #df_qline.plot(kind='line', x='x', y='y', color='red', ax=ax, legend=False, style='--')


    if save:
        filename = kwargs.get('filename') if kwargs.get('filename') else f"manhattan_{len(stats_results)}_{str(datetime.date.today())}.png"
        plt.savefig(filename)
        if verbose == True:
            print(f"saved {filename}")
    if verbose == True:
        plt.show()
    else:
        plt.close(fig)

stats_results.to_csv(filename)

This function writes the pandas DataFrame output of diff_meth_pos() to a CSV file
named by the user. The DataFrame has a row for every successfully tested probe
and columns with different regression statistics as follows:
        - regression coefficient
        - lower limit of the coefficient's 95% confidence interval
        - upper limit of the coefficient's 95% confidence interval
        - standard error
        - p-value
        - q-value (p-values corrected for multiple testing using the Benjamini-Hochberg FDR method)

Inputs and Parameters:

    stats_results: A pandas DataFrame output by the function diff_meth_pos().
    filename: A string that will be used to name the resulting .CSV file.

Returns:
    Writes a CSV file, but does not directly return an object.
    The CSV will include the DataFrame column names as headers and the index
    of the DataFrame as row names for each probe.


def mantest(debug=True):
    import methylize as m
    import pandas as pd
    meth_data = pd.read_pickle('data/GSE69852_beta_values.pkl').transpose()
    pheno_data = [0,0,1,0,1,0] # ["0","1","0","1","0","1"]
    sample = meth_data.sample(5000,axis=1)
    r1 = m.diff_meth_pos(sample, pheno_data, 'linear', export=False, debug=debug)
    r2 = m.diff_meth_pos(sample, pheno_data, 'linear', export=False, debug=debug, solver='statsmodels_OLS')
    for col in r2.columns:
        print(col, (r2[col] - r1[col]).mean(), (r1[col]/r2[col]).mean(), (r1[col]/r2[col]).std())
    return r1,r2

    m.volcano_plot(r1)
    #return res
    m.manhattan_plot(res, '450k', palette='Gray') # fontsize=10, fwer=0.001, save=False,


def test3(what='disease status', debug=False):
    import methylize as m
    import pandas as pd
    from pathlib import Path
    path = Path('/Volumes/LEGX/GEO/GSE85566/GPL13534/')
    beta = pd.read_pickle(Path(path,'beta_values.pkl'))
    pheno = pd.read_pickle(Path(path,'GSE85566_GPL13534_meta_data.pkl'))[['disease status','gender']] # 'gender' or 'disease status'
    sample = beta.sample(2000)
    r1 = m.diff_meth_pos(sample, pheno, debug=True, column='disease status', covariates=True, solver='statsmodels_GLM', impute='average')
    m.manhattan_plot(r1, '450k', save=False, explore=True)


    r2 = m.diff_meth_pos(sample, pheno, debug=True, column='disease status', covariates=True, impute='average')
    m.manhattan_plot(r2, '450k', save=True, filename='test-r2.png')
    return r1,r2


    pheno3 = pd.read_pickle(Path(path,'GSE85566_GPL13534_meta_data.pkl'))[['disease status','gender','age']]
    r2 = m.diff_meth_pos(sample, pheno3, debug=True, column='disease status', covariates=True)
    r3 = m.diff_meth_pos(sample, pheno, debug=True, column='disease status', covariates=None)
    pheno4 = pheno3.copy()
    pheno4['dummy'] = 5.0 # testing zero variance case
    r4 = m.diff_meth_pos(sample, pheno4, debug=True, column='disease status', covariates=True)
    m.manhattan_plot(r1, '450k', save=True, filename='test-r1.png')
    m.manhattan_plot(r2, '450k', save=True, filename='test-r2.png')
    m.manhattan_plot(r3, '450k', save=True, filename='test-r3.png')
    m.manhattan_plot(r4, '450k', save=True, filename='test-r4.png')
    return r4

    #return r1, r2, r3
    #result = m.diff_meth_pos(sample, pheno, 'logistic', solver='statsmodels_GLM')
    m.manhattan_plot(result, '450k', save=True, filename='testglm.png')

    #m.volcano_plot(result, adjust=False, cutoff=(-0.2, 0.2))




"""

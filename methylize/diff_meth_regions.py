import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats
#from joblib import Parallel, delayed, cpu_count
import matplotlib.pyplot as plt
import matplotlib # color maps
# app
from .helpers import color_schemes, create_probe_chr_map, create_mapinfo

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def bumphunter(df, pheno, array_type, **kwargs):
    """Discover differentially methylated genomic regions (DMRs), ported from `minfi`'s bumpHunter function.

    inputs:
    -------
    df:
        A dataframe of beta values or M values. Samples in one axis and probe names in the other. It will be
        transposed to the proper orientation if needed.
    pheno:
        Phenotype data (list or array-like) in the same order and matching the same length as the methylation data.
    array_type:
        specify one of {'450k','epic', 'epic+', 'mouse'} so that the function will load the appropriate
        probe-to-genomic locus map from manifest.

.. How it works:
    The first set of commands within the bumphunterEngine check which arguments were specified when the bumphunter
    function was called, check if they're in the correct format, and produce informative warnings if any input is
    incorrect.
    Then the bumphunterEngine calls several functions from R's foreach package to see if multiple cores were registered
    before bumphunter was called. This sets up the makes it possible for the more time-consuming bumphunterEngine
    functions to process in parallel as coded.
    The first backend function bumphunterEngine calls is clusterMaker, which assigns genomic locations to sets of
    clusters if they are within maxGap of one another (if clusters were not already pre-assigned in one of the
    function arguments).

.. note:
    Parallel processing has NOT been addressed yet in this python implementation
    """
    array_types = {'450k','epic', 'epic+', 'mouse'}
    if array_type not in array_types:
        raise ValueError(f"array_type must be one of these: {array_types}")
    if len(df.columns) > 27000 and len(df.index) < 27000:
        df = df.transpose()
        if kwargs.get('verbose') == True:
            logger.info(f"Your data was transposed: {df.shape}")

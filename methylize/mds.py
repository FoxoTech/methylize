# about
# provides an interface for combining multiple datasets (or splitting one dataset by assoc meta data)
# into an overlay MDS plot.

import pandas as pd
import numpy as np
# app
import methQC

def combine_mds(*args, **kwargs):
    """ pass in any number of dataframes, and it will combine them into one mds plot. """
    list_of_dfs = list(args)
    silent=kwargs.get('silent', False)
    verbose=kwargs.get('verbose', True)
    
    dfs = pd.DataFrame()
    sample_source = {}

    # TRACK each df's samples for color coding prior to merging
    plots = []
    for idx, df in enumerate(list_of_dfs):
        if df.shape[1] > df.shape[0]: # put probes in rows
            df = df.transpose()    
        for sample in df.columns:
            sample_source[sample] = idx

        # PLOT separate MDS
        try:
            plt = methQC.beta_mds_plot(dfs, save=True, verbose=False, silent=True, return_plot_obj=True)
            plots.append(plt)
        except:
            pass        
        
        # MERGE
        #dfs = pd.concat([dfs, df], sort=True, axis=1, join='outer', ignore_index=False)
        #print(dfs.shape)
    return plots

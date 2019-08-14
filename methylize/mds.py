# about
# provides an interface for combining multiple datasets (or splitting one dataset by assoc meta data)
# into an overlay MDS plot.
import logging
import os
from tqdm import tqdm
import datetime
# package
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import MDS
import pandas as pd
import numpy as np
# app
import methQC

__all__ = ['combine_mds']

LOGGER = logging.getLogger(__name__)

def combine_mds(*args, **kwargs):
    """
    how it works
    ------------
    Use this function on multiple dataframes to combine datasets, or to visualize
    parts of the same dataset in separate colors. It is a wrapper of `methQC.beta_mds_plot()` and applies
    multidimensional scaling to cluster similar samples based on patterns in probe values, as well as identify
    possible outlier samples (and exclude them).
    
    - combine datasets,
    - run MDS,
    - see how each dataset (or subset) overlaps with the others on a plot,
    - exclude outlier samples based on a composite cutoff box (the average bounds of the component data sets)
    - calculate the percent of data excluded from the group

    inputs
    ------
        - *args: pass in any number of pandas dataframes, and it will combine them into one mds plot.
        - alternatively, you may pass in a list of filepaths as strings, and it will attempt to load these files as pickles. 
        but they must be pickles of pandas dataframes
    
    optional keyword arguments
    --------------------------
    - silent: (default False)
        (automated processing mode) 
        if True, suppresses most information and avoids prompting user for anything.
        silent mode processes data but doesn't show the plot.    
    - save: (default False)
        if True, saves the plot png to disk.
    - verbose: (default False)
        if True, prints extra debug information to screen or logger.
        
    returns
    ------  
        nothing returned (currently)
        - default: list of samples retained or excluded
        - option: a list of pyplot subplot objects         
        - TODO: one dataframe of the retained samples.
    """
    
    # check if already dataframes, or are they strings?
    list_of_dfs = list(args)    
    if any([not isinstance(item, pd.DataFrame) for item in list_of_dfs]):
        if set([type(item) for item in list_of_dfs]) == {str}:
            try:
                list_of_dfs = _load_data(list_of_dfs)
            except Exception as e:
                raise FileNotFoundError ("Either your files don't exist, or they are not pickled: {0}".format(e))
    # kwargs
    save = kwargs.get('save', False)
    silent = kwargs.get('silent', False)
    verbose = kwargs.get('verbose', True)
    PRINT = print if verbose else _noprint
    
    
    # data to combine
    dfs = pd.DataFrame()    
    # OPTIONAL OUTPUT: TRACK each source df's samples for plot color coding prior to merging
    # i.e. was this sample included or excluded at end?
    sample_source = {}    
    # subplots: possibly useful - list of subplot objects within the figure, and their metadata.
    subplots = []
    # track x/y ranges to adjust plot area later
    xy_lims = []
    
    # PROCESS MDS
    fig = None    
    for idx, df in enumerate(list_of_dfs):
        if df.shape[1] > df.shape[0]: # put probes in rows
            df = df.transpose()    
        for sample in df.columns:
            sample_source[sample] = idx

        # PLOT separate MDS
        PRINT(idx, fig)
        #-- only draw last iteration: draw_box = True if idx == len(list_of_dfs)-1 else False
        #-- draw after complete, using dimensions provided
        try:
            fig = mds_plot(df, color_num=idx, save=save, verbose=verbose, silent=silent, return_plot_obj=True, fig=fig, draw_box=True)
            subplots.append(fig)
            xy_lims.append( (fig.axes[0].get_xlim(), fig.axes[0].get_ylim()) ) # (x_range, y_range)
        except Exception as e:
            PRINT(e)      

    #set max range
    x_range_min = min(item[0][0] for item in xy_lims)
    x_range_max = max(item[0][1] for item in xy_lims)        
    y_range_min = min(item[1][0] for item in xy_lims)
    y_range_max = max(item[1][1] for item in xy_lims)
    fig.axes[0].set_xlim([x_range_min, x_range_max])
    fig.axes[0].set_ylim([y_range_min, y_range_max])
    PRINT(int(x_range_min), int(x_range_max), int(y_range_min), int(y_range_max))    
    if silent:
        plt.show()
    plt.close()

    # PART 2 - calculate the average MDS QC score
    #1 run the loop. calc avg MDS boundary box for group.
    #2 rerun the loop with this as the specified boundary.
    #3 calculate percent retained per dataset
    #4 overall QC score is avg percent retained per dataset.
    avg_x_range_min = int(sum(item[0][0] for item in xy_lims)/len(xy_lims))
    avg_x_range_max = int(sum(item[0][1] for item in xy_lims)/len(xy_lims))       
    avg_y_range_min = int(sum(item[1][0] for item in xy_lims)/len(xy_lims))
    avg_y_range_max = int(sum(item[1][1] for item in xy_lims)/len(xy_lims))
    xy_lim = ((avg_x_range_min, avg_x_range_max), (avg_y_range_min, avg_y_range_max))
    PRINT('AVG',xy_lim)
    fig=None
    for idx, df in enumerate(list_of_dfs):
        if df.shape[1] > df.shape[0]: # put probes in rows
            df = df.transpose()    
        PRINT(idx, fig)
        fig = mds_plot(df, color_num=idx, save=save, verbose=verbose, silent=silent, return_plot_obj=True, fig=fig, draw_box=True, xy_lim=xy_lim)
    fig.axes[0].set_xlim([x_range_min, x_range_max])
    fig.axes[0].set_ylim([y_range_min, y_range_max])

    # https://stackoverflow.com/questions/32213889/get-positions-of-points-in-pathcollection-created-by-scatter
    #print(len(fig.axes[0].collections)) # collections has data in every 4th list item.
    # items in the collections after the main data set are the excluded points
    all_coords = []    
    retained = []
    excluded = []
    for i in range(0, 4*len(list_of_dfs), 1):
        DD = fig.axes[0].collections[i]
        DD.set_offset_position('data')
        #print(DD.get_offsets())
        all_coords.extend(DD.get_offsets().tolist())
        #print(i, len(all_coords))
        if i % 4 == 0: # 0, 4, 8 -- this is the first data set applied to plot.
            retained.extend( DD.get_offsets().tolist() )
        if i % 4 == 1: # 1, 5, 9, etc -- this is the second data set applied to plot.
            excluded.extend( DD.get_offsets().tolist() )
    if verbose:            
        PRINT( round(100*len(retained) / (len(retained) + len(excluded))), '% retained overall')

    # NOT USED: calculate percent excluded across datasets: -- not needed.
    def within(coord, xy_lim):
        # coord is [x,y] and xy_lim is ((xmin,xmax), (ymin,ymax))
        # if either x or y value is outside the limits, it is not within and returns False.
        if coord[0] < xy_lim[0][0] or coord[0] > xy_lim[0][1]:
            return 0
        elif coord[1] < xy_lim[1][0] or coord[1] > xy_lim[1][1]:
            return 0
        return 1
    
    # uses xy_lim, all_coords
    # print(xy_lim)
    # print(sum([within(coord, xy_lim) for coord in all_coords])/len(all_coords),'retained overall')
    if silent:
        plt.show()
    plt.close()


def mds_plot(df, color_num=0, filter_stdev=2, verbose=True, save=False, silent=False, return_plot_obj=True, fig=None, draw_box=False, xy_lim=None):
    """ based on methQC's plot, but this version takes a plt object and adds to it from a wrapper function.
    
    
    1 needs to read the manifest file for the array, or at least a list of probe names to exclude/include.
        manifest_file = pd.read_csv('/Users/nrigby/GitHub/stp-prelim-analysis/working_data/CombinedManifestEPIC.manifest.CoreColumns.csv')[['IlmnID', 'CHR']]
        probe_names_no_sex_probes = manifest_file.loc[manifest_file['CHR'].apply(lambda x: x not in ['X', 'Y', np.nan]), 'IlmnID'].values
        probe_names_sex_probes = manifest_file.loc[manifest_file['CHR'].apply(lambda x: x in ['X', 'Y']), 'IlmnID'].values

    df_no_sex_probes = df[probe_names_no_sex_probes]
    df_no_sex_probes.head()

    Arguments
    ---------
    df
        dataframe of beta values for a batch of samples (rows are probes; cols are samples)
    filter_stdev
        a value (unit: standard deviations) between 0 and 3 (typically) that represents
        the fraction of samples to include, based on the standard deviation of this batch of samples.
        So using the default value of 1.5 means that all samples whose MDS-transformed beta sort_values
        are within +/- 1.5 standard deviations of the average beta are retained in the data returned.

    Options
    --------
    silent
        if running from command line in an automated process, you can run in `silent` mode to suppress any user interaction.
        In this case, whatever `filter_stdev` you assign is the final value, and a file will be processed with that param.
        Silent also suppresses plots (images) from being generated. only files are returned.

    returns
    -------
        returns a filtered dataframe.
        if `return_plot_obj` is True, it returns the plot, for making overlays in methylize.

    requires
    --------
        pandas, numpy, pyplot, sklearn.manifold.MDS"""
    # ensure "long format": probes in rows and samples in cols. This is how methpype returns data.
    if df.shape[1] < df.shape[0]:
        ## methQC needs probes in rows and samples in cols. but MDS needs a wide matrix.
        df = df.copy().transpose() # don't overwrite the original
        if verbose:
            print("Your data needed to be transposed (df = df.transpose()).")
            LOGGER.info("Your data needed to be transposed (df = df.transpose()).")
    if verbose == True:
        print(df.shape)
        df.head()
        LOGGER.info('DataFrame has shape: {0}'.format(df.shape))
        print("Making sure that probes are in columns (the second number should be larger than the first).")
        LOGGER.info("Making sure that probes are in columns (the second number should be larger than the first).")
        # before running this, you'd typically exclude probes.
        print("Starting MDS fit_transform. this may take a while.")
        LOGGER.info("Starting MDS fit_transform. this may take a while.")

    #df = drop_nan_probes(df, silent=silent, verbose=verbose)

    # CHECK for missing probe values NaN
    missing_probe_counts = df.isna().sum()
    total_missing_probes = sum([i for i in missing_probe_counts])/len(df) # sum / columns
    if sum([i for i in missing_probe_counts]) > 0:
        df = df.dropna()
        if verbose == True:
            print("We found {0} probe(s) were missing values and removed them from MDS calculations.".format(total_missing_probes))
        if silent == False:
            LOGGER.info("{0} probe(s) were missing values removed from MDS calculations.".format(total_missing_probes))


    mds = MDS(n_jobs=-1, random_state=1, verbose=1)
    #n_jobs=-1 means "use all processors"
    mds_transformed = mds.fit_transform(df.values)
    # pass in df.values (a np.array) instead of a dataframe, as it processes much faster.
    # old range is used for plotting, with some margin on outside for chart readability
    PSF = 1.2 # plot_scale_fator -- an empirical number to stretch the plot out and show clusters more easily.
    if df.shape[0] < 40:
        DOTSIZE = 16
    elif 40 < df.shape[0] < 60:
        DOTSIZE = 14
    elif 40 < df.shape[0] < 60:
        DOTSIZE = 12
    elif 60 < df.shape[0] < 80:
        DOTSIZE = 10
    elif 80 < df.shape[0] < 100:
        DOTSIZE = 8
    elif 100 < df.shape[0] < 300:
        DOTSIZE = 7
    else:
        DOTSIZE = 5
    old_X_range = [min(mds_transformed[:, 0]), max(mds_transformed[:, 0])]
    old_Y_range = [min(mds_transformed[:, 1]), max(mds_transformed[:, 1])]
    #old_X_range = [old_X_range[0] - PSF*old_X_range[0], old_X_range[1] + PSF*old_X_range[1]]
    #old_Y_range = [old_Y_range[0] - PSF*old_Y_range[0], old_Y_range[1] + PSF*old_Y_range[1]]
    x_std, y_std = np.std(mds_transformed,axis=0)
    x_avg, y_avg = np.mean(mds_transformed,axis=0)

    adj = filter_stdev #(1.5)
    ##########
    if verbose == True:
        print("""You can now remove outliers based on their transformed beta values
 falling outside a range, defined by the sample standard deviation.""")
    while True:
        df_indexes_to_retain = []
        df_indexes_to_exclude = []
        minX = round(x_avg - adj*x_std)
        maxX = round(x_avg + adj*x_std)
        minY = round(y_avg - adj*y_std)
        maxY = round(y_avg + adj*y_std)
        if type(xy_lim) in (list,tuple): # ((xmin, xmax), (ymin, ymax))
            # xy_lim is a manual override preset boundary cutoff range for MDS filtering.
            minX = xy_lim[0][0]
            maxX = xy_lim[0][1]
            minY = xy_lim[1][0]
            maxY = xy_lim[1][1]
            
        if verbose == True:
            print('Your acceptable value range: x=({0} to {1}), y=({2} to {3}).'.format(
                minX, maxX,
                minY, maxY
            ))
        md2 = [] # md2 are the dots that fall inside the cutoff.
        for idx,row in enumerate(mds_transformed):
            if minX <= row[0] <= maxX and minY <= row[1] <= maxY:
                md2.append(row)
                df_indexes_to_retain.append(idx)
            else:
                df_indexes_to_exclude.append(idx)
        #pandas style: mds2 = mds_transformed[mds_transformed[:, 0] == class_number[:, :2]
        # this is a np array, not a df. Removing all dots that are retained from the "exluded" data set (mds_transformed)
        mds_transformed = np.delete(mds_transformed, [df_indexes_to_retain], axis=0)                
        md2 = np.array(md2)

        if not fig:
            # DO THIS PART ONCE
            fig = plt.figure(figsize=(12, 9))
            plt.title('MDS Plot of betas from methylation data')
            plt.grid()
        # ADD TO EXISTING FIG... up to 24 colors (repeated 3x for 72 total)
        COLORSET = dict(enumerate(['xkcd:blue', 'xkcd:green', 'xkcd:coral', 'xkcd:lightblue', 'xkcd:magenta', 'xkcd:goldenrod', 'xkcd:plum', 'xkcd:beige',
                                   'xkcd:orange', 'xkcd:orchid', 'xkcd:silver', 'xkcd:purple', 'xkcd:pink', 'xkcd:teal', 'xkcd:tomato', 'xkcd:yellow',
                                   'xkcd:olive', 'xkcd:lavender', 'xkcd:indigo', 'xkcd:black', 'xkcd:azure', 'xkcd:brown', 'xkcd:aquamarine', 'xkcd:darkblue',
                                   
                                   'xkcd:blue', 'xkcd:green', 'xkcd:coral', 'xkcd:lightblue', 'xkcd:magenta', 'xkcd:goldenrod', 'xkcd:plum', 'xkcd:beige',
                                   'xkcd:orange', 'xkcd:orchid', 'xkcd:silver', 'xkcd:purple', 'xkcd:pink', 'xkcd:teal', 'xkcd:tomato', 'xkcd:yellow',
                                   'xkcd:olive', 'xkcd:lavender', 'xkcd:indigo', 'xkcd:black', 'xkcd:azure', 'xkcd:brown', 'xkcd:aquamarine', 'xkcd:darkblue',
                                   
                                   'xkcd:blue', 'xkcd:green', 'xkcd:coral', 'xkcd:lightblue', 'xkcd:magenta', 'xkcd:goldenrod', 'xkcd:plum', 'xkcd:beige',
                                   'xkcd:orange', 'xkcd:orchid', 'xkcd:silver', 'xkcd:purple', 'xkcd:pink', 'xkcd:teal', 'xkcd:tomato', 'xkcd:yellow',
                                   'xkcd:olive', 'xkcd:lavender', 'xkcd:indigo', 'xkcd:black', 'xkcd:azure', 'xkcd:brown', 'xkcd:aquamarine', 'xkcd:darkblue'])) 
        ax = fig.add_subplot(1,1,1)

        ax.scatter(md2[:, 0], md2[:, 1], s=DOTSIZE, c=COLORSET.get(color_num)) # RETAINED        
        ax.scatter(mds_transformed[:, 0], mds_transformed[:, 1], s=DOTSIZE, c='xkcd:ivory', edgecolor='black', linewidth='0.2',) # EXCLUDED

        #ax = fig.add_subplot(1,1,1, label='label'+str(color_num)) -- failed
        #ax = fig.subplots(1,1, sharex=True, sharey=True) -- failed

        x_range_min = PSF*old_X_range[0] if PSF*old_X_range[0] < minX else PSF*minX
        x_range_max = PSF*old_X_range[1] if PSF*old_X_range[1] > maxX else PSF*maxX
        y_range_min = PSF*old_Y_range[0] if PSF*old_Y_range[0] < minY else PSF*minY
        y_range_max = PSF*old_Y_range[1] if PSF*old_Y_range[1] > maxY else PSF*maxY
        ax.set_xlim([x_range_min, x_range_max])
        #ax.set_xlim([-300, 300])
        ax.set_ylim([y_range_min, y_range_max])
        #ax.set_ylim([-300, 300])
        #print(int(x_range_min), int(x_range_max), int(y_range_min), int(y_range_max))
        
        if draw_box:
            ax.vlines([minX, maxX], minY, maxY, color=COLORSET.get(color_num,'red'), linestyle=':')
            ax.hlines([minY, maxY], minX, maxX, color=COLORSET.get(color_num,'red'), linestyle=':')

        if return_plot_obj == True:
            return fig
        elif silent == True:
            # take the original dataframe (df) and remove samples that are outside the sample thresholds, returning a new dataframe
            df.drop(df.index[df_indexes_to_exclude], inplace=True)
            image_name = df.index.name or 'beta_mds_n={0}_p={1}'.format(len(df.index), len(df.columns)) # np.size(df,0), np.size(md2,1)
            outfile = '{0}_s={1}_{2}.png'.format(image_name, filter_stdev, datetime.date.today())
            plt.savefig(outfile)
            LOGGER.info("Saved {0}".format(outfile))
            plt.close(fig)
            # returning DataFrame in original structure: rows are probes; cols are samples.
            return df  # may need to transpose this first.
        else:
            plt.show()

        ########## BEGIN INTERACTIVE MODE #############
        print("Original samples {0} vs filtered {1}".format(mds_transformed.shape, md2.shape))
        print('Your scale factor was: {0}'.format(adj))
        adj = input("Enter new scale factor, <enter> to accept and save:")
        if adj == '':
            break
        try:
            adj = float(adj)
        except ValueError:
            print("Not a valid number. Type a number with a decimal value, or Press <enter> to quit.")
            continue

    # save file. return dataframe.
    fig = plt.figure(figsize=(12, 9))
    plt.title('MDS Plot of betas from methylation data')
    plt.scatter(mds_transformed[:, 0], mds_transformed[:, 1], s=5)
    plt.scatter(md2[:, 0], md2[:, 1], s=5, c='red')
    plt.xlim(old_X_range)
    plt.ylim(old_Y_range)
    plt.xlabel('X')
    plt.ylabel('Y')
    # take the original dataframe (df) and remove samples that are outside the sample thresholds, returning a new dataframe
    df_out = df.drop(df.index[df_indexes_to_exclude]) # inplace=True will modify the original DF outside this function.

    # UNRESOLVED BUG.
    # was getting 1069 samples back from this; expected 1076. discrepancy is because
    # pre_df_excl = len(df.index[df_indexes_to_exclude])
    # unique_df_excl = len(set(df.index[df_indexes_to_exclude]))
    # print(pre_df_excl, unique_df_excl)

    prev_df = len(df)
    if save:
        image_name = df.index.name or 'beta_mds_n={0}_p={1}'.format(len(df.index), len(df.columns)) # np.size(df,0), np.size(md2,1)
        outfile = '{0}_s={1}_{2}.png'.format(image_name, filter_stdev, datetime.date.today())
        plt.savefig(outfile)
        if verbose:
            print("Saved {0}".format(outfile))
            LOGGER.info("Saved {0}".format(outfile))
    plt.close(fig) # avoids displaying plot again in jupyter.
    # returning DataFrame in original structure: rows are probes; cols are samples.
    return df_out #, df_indexes_to_exclude  # may need to transpose this first.

def _load_data(filepaths):    
    dfs = []
    for ff in filepaths:
        df = pd.read_pickle(ff)
        dfs.append(df)
    return dfs

def _noprint(message):
    """ a helper function to suppress print() if not verbose mode. """
    pass
    

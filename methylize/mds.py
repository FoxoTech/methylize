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


def combine_mds(*args, **kwargs):
    """ pass in any number of dataframes, and it will combine them into one mds plot. """
    list_of_dfs = list(args)
    #silent=kwargs.get('silent', False)
    #verbose=kwargs.get('verbose', True)
    
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
        fig = None
        try:
            fig = mds_plot(df, fig=fig, color_num=idx, save=True, verbose=False, silent=True, return_plot_obj=True)
            plots.append(fig)
        except Exception as e:
            print(e)
            pass        
        # MERGE
        #dfs = pd.concat([dfs, df], sort=True, axis=1, join='outer', ignore_index=False)
        #print(dfs.shape)
    plots[-1].show()
    return plots


def mds_plot(df, fig=None, color_num=0, filter_stdev=1.5, verbose=True, save=False, silent=False, return_plot_obj=True):
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
    PSF = 2 # plot_scale_fator -- an empirical number to stretch the plot out and show clusters more easily.
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
        df_indexes_to_exclude = []
        minX = round(x_avg - adj*x_std)
        maxX = round(x_avg + adj*x_std)
        minY = round(y_avg - adj*y_std)
        maxY = round(y_avg + adj*y_std)
        if verbose == True:
            print('Your acceptable value range: x=({0} to {1}), y=({2} to {3}).'.format(
                minX, maxX,
                minY, maxY
            ))
        md2 = []
        for idx,row in enumerate(mds_transformed):
            if minX <= row[0] <= maxX and minY <= row[1] <= maxY:
                md2.append(row)
            else:
                df_indexes_to_exclude.append(idx)
            #pandas style: mds2 = mds_transformed[mds_transformed[:, 0] == class_number[:, :2]
        md2 = np.array(md2)
        if not fig:
            # DO THIS PART ONCE
            fig = plt.figure(figsize=(12, 9))
            plt.title('MDS Plot of betas from methylation data')
            plt.grid()
        # ADD TO EXISTING FIG...
        ax = fig.add_subplot(1,1,1)        
        ax.scatter(mds_transformed[:, 0], mds_transformed[:, 1], s=DOTSIZE)
        COLORSET = dict(enumerate({'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'})) # 0-7 allowed
        ax.scatter(md2[:, 0], md2[:, 1], s=5, c=COLORSET.get(color_num,'red'))

        x_range_min = PSF*old_X_range[0] if PSF*old_X_range[0] < minX else PSF*minX
        x_range_max = PSF*old_X_range[1] if PSF*old_X_range[1] > maxX else PSF*maxX
        y_range_min = PSF*old_Y_range[0] if PSF*old_Y_range[0] < minY else PSF*minY
        y_range_max = PSF*old_Y_range[1] if PSF*old_Y_range[1] > maxY else PSF*maxY
        ax.xlim([x_range_min, x_range_max])
        ax.ylim([y_range_min, y_range_max])
        ax.vlines([minX, maxX], minY, maxY, color='red', linestyle=':')
        ax.hlines([minY, maxY], minX, maxX, color='red', linestyle=':')
        #line = mlines.Line2D([minX, minX], [minY, minY], color='red', linestyle='--')
        #ax = fig.add_subplot(111)
        #ax.add_line(line)

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


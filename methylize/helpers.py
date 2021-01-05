import logging
import matplotlib.pyplot as plt
import matplotlib # color maps
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import chain
from importlib import resources # py3.7+
from .progress_bar import * # context tqdm

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

__all__ = ['load']

pkg_namespace = 'methylize.data'
probe2chr = None
probe2map = None
def load_probe_chr_map():
    """ runs inside manhattan plot, and only needed there, but useful to load once if function called multiple times """
    global probe2chr
    if probe2chr != None:
        return
    # maps probes to chromosomes for all known probes in major array types.
    import pickle
    from pathlib import Path
    #with open(Path(Path(__file__).parent,'data','probe2chr.pkl'),'rb') as f:
    #    probe2map = pickle.load(f)
    with resources.path(pkg_namespace, 'probe2chr.pkl') as probe2chr_filepath:
        with open(probe2chr_filepath,'rb') as f:
            probe2map = pickle.load(f)
        # structure is dataframe with 'CGidentifier, CHR, MAPINFO' columns -- bumphunter uses MAPINFO (chromosome position of probes)
    # dict for volcano plots, with a sortable order for chart
    probes = probe2map[['CGidentifier','CHR']].to_dict('records')
    probe2chr = {probe['CGidentifier']:f"CHR-0{probe['CHR']}" if probe['CHR'] not in ('X','Y') and type(probe['CHR']) is str and int(probe['CHR']) < 10 else f"CHR-{probe['CHR']}" for probe in probes}
    #OLD probe2chr = {k:f"CH-0{v}" if v not in ('X','Y') and type(v) is str and int(v) < 10 else f"CH-{v}" for k,v in probes.items()}
load_probe_chr_map()

color_schemes = {}
def load_color_schemes():
    global color_schemes
    if color_schemes != {}:
        return
    color_schemes = {cmap: matplotlib.cm.get_cmap(cmap) for cmap in
        ['Pastel1', 'Pastel2', 'Paired', 'Accent',
        'Dark2', 'Set1', 'Set2', 'Set3',
        'tab10', 'tab20', 'tab20b', 'tab20c']
    }
    color_schemes['Gray'] = matplotlib.colors.ListedColormap(['whitesmoke','lightgray','silver','darkgray','gray','dimgray','black'])
    color_schemes['Gray2'] = matplotlib.colors.ListedColormap(['silver','gray'])
    color_schemes['Gray3'] = matplotlib.colors.ListedColormap(['darkgrey','black'])
    color_schemes['Volcano'] = matplotlib.colors.ListedColormap(['tab:red','tab:blue','silver'])
    color_schemes['default'] = matplotlib.colors.ListedColormap(['mistyrose', 'navajowhite', 'palegoldenrod', 'yellowgreen', 'mediumseagreen', 'powderblue', 'skyblue',  'lavender', 'plum', 'palevioletred'])
load_color_schemes()


def map_to_genome(df, rgset):
    """Maps dataframe to genome locations
    Parameters
    ----------
    df: dataframe
            Dataframe containing methylation, unmethylation, M or Beta
            values for each sample at each site
    rgset: rg channel set instance
            RG channel set instance related to provided df
    Returns
    -------
    df: dataframe
            Dataframe containing the original values with the addition
            of genomic locations for each site
    """
    # This (mapToGenome) was in an old copy of methpype, but moved to methylize where it might be used.
    if 'Name' in df.columns:
        mani = rgset.manifest
        chromosomes = dict(
            zip(list(mani['Name'].values), list(mani['CHR'].values)))
        strands = dict(
            zip(list(mani['Name'].values), list(mani['Strand'].values)))
        build = dict(zip(list(mani['Name'].values),
                         list(mani['Genome_Build'].values)))
        mapinfo = dict(
            zip(list(mani['Name'].values), list(mani['MAPINFO'].values)))
        df['Chr'] = df['Name'].map(chromosomes)
        df['Strand'] = df['Name'].map(strands)
        df['GenomeBuild'] = df['Name'].map(build)
        df['MapInfo'] = df['Name'].map(mapinfo)
        return df
    else:
        print("No 'Name' column in dataframe.")
        return

def readManifest(array):
    """DEPRECRATED VERSION -- Returns Illumina manifest for array type
    Parameters
    ----------
    array: str
        String specifying the type of Illumina Methylation Array
    Returns
    -------
    manifest: dataframe
        Dataframe containing Illumina Human Methylation Array manifest
    """
    # Copy of manifest-reading code from old version of methpype
    downloadManifest(array)
    if array == 'CustomArray':
        manifest = pd.read_csv(os.path.expanduser(
            "~/.methpype_manifest_files/CombinedManifestEPIC.manifest.CoreColumns.csv.gz"))
        # TEMPORARY - Remove missing probes
        manifest = manifest[manifest['AddressA_ID'] != 9614306]
        manifest = manifest[manifest['AddressA_ID'] != 9637526]
        manifest = manifest[manifest['AddressA_ID'] != 3680876]
        manifest = manifest[manifest['AddressB_ID'] != 60646183]
    elif array == 'IlluminaHumanMethylation450k':
        #manifest = pd.read_csv(os.path.expanduser("~/.methpype_manifest_files/HumanMethylation450_15017482_v1-2.CoreColumns.csv.gz"))
        manifest = pd.read_csv("~/HumanMethylation450_15017482_v1-2.csv")
    elif array == 'IlluminaHumanMethylationEPIC':
        manifest = pd.read_csv(os.path.expanduser(
            "~/.methpype_manifest_files/MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz"))
    manifest = manifest.rename({'AddressA_ID': 'AddressA', 'AddressB_ID': 'AddressB', 'Infinium_Design_Type': 'Type',
                                'Color_Channel': 'Color', 'IlmnID': 'CtrlAddress', 'Name': 'Name'}, axis='columns')
    manifest.dropna(how='all', inplace=True)
    manifest = flagProbeTypes(manifest)
    # CURRENTLY UNUSED CODE
    # KEEP FOR FUTURE ITERATION WHERE PROBE SEQUENCES CAN BE MAPPED BACK
    # # determine nCpGs
    # nCpGs = {}
    # for name,A,B,T in zip(manifest['Name'],manifest['ProbeSeqA'],manifest['ProbeSeqB'],manifest['Type']):
    #   if T == 'I':
    #       nCpGs[name] = int(B.count("CG")-1)
    #   elif T == 'II':
    #       nCpGs[name] = int(A.count('R'))
    #   else:
    #       nCpGs[name] = int(0)
    # # add to dataframe
    # manifest['nCpG'] = manifest['Name'].map(nCpGs)
    # determine types
    return manifest


def load_both(filepath='.', format='beta_values', file_stem='', verbose=False, silent=False):
    """Loads any pickled files in the given filepath that match specified format,
    plus the associated meta data frame. Returns TWO objects (data, meta) as dataframes for analysis.

    If meta_data files are found in multiple folders, it will read them all and try to match to the samples
    in the beta_values pickles by sample ID.

Arguments:
    filepath:
        Where to look for all the pickle files of processed data.

    format:
        'beta_values', 'm_value', or some other custom file pattern.

    file_stem (string):
        By default, methylprep process with batch_size creates a bunch of generically named files, such as
        'beta_values_1.pkl', 'beta_values_2.pkl', 'beta_values_3.pkl', and so on. IF you rename these or provide
        a custom name during processing, provide that name here.
        (i.e. if your pickle file is called 'GSE150999_beta_values_X.pkl', then your file_stem is 'GSE150999_')

    verbose:
        outputs more processing messages.
    silent:
        suppresses all processing messages, even warnings.
    """
    meta_files = list(Path(filepath).rglob(f'*_meta_data.pkl'))
    multiple_metas = False
    partial_meta = False
    if len(meta_files) > 1:
        LOGGER.info(f"Found several meta_data files; attempting to match each with its respective beta_values files in same folders.")
        multiple_metas = True # this will skip the df-size-match check below.
        ### if multiple_metas, combine them into one dataframe of meta data with
        ### all samples rows and tags in columns
        # Note: this approach assumes that:
        #    (a) the goal is a row-wise concatenation (i.e., axis=0) and
        #    (b) all dataframes share the same column names.
        frames = [pd.read_pickle(pkl) for pkl in meta_files]
        meta_tags = frames[0].columns
        # do all match?
        meta_sets = set()
        for frame in frames:
            meta_sets |= set(frame.columns)
        if meta_sets != set(meta_tags):
            LOGGER.warning(f'Columns in sample sheet meta data files does not match for these files and cannot be combined:'
                           f'{[str(i) for i in meta_files]}')
            meta = pd.read_pickle(meta_files[0])
            partial_meta = True
        else:
            meta = pd.concat(frames, axis=0, sort=False)
            # need to check whether there are multiple samples for each sample name. and warn.

    if len(meta_files) == 1:
        meta = pd.read_pickle(meta_files[0])
    elif multiple_metas:
        if partial_meta:
            LOGGER.info("Multiple meta_data found. Only loading the first file.")
        LOGGER.info(f"Loading {len(meta.index)} samples.")
    else:
        LOGGER.info("No meta_data found.")
        meta = pd.DataFrame()

    data_df = load(filepath=filepath,
        format=format,
        file_stem=file_stem,
        verbose=verbose,
        silent=silent
        )

    ### confirm the Sample_ID in meta matches the columns (or index) in data_df.
    check = False
    if 'Sample_ID' in meta.columns:
        if len(meta['Sample_ID']) == len(data_df.columns) and all(meta['Sample_ID'] == data_df.columns):
            data_df = data_df.transpose() # samples should be in index
            check = True
        elif len(meta['Sample_ID']) == len(data_df.index) and all(meta['Sample_ID'] == data_df.index):
            check = True
        # or maybe the data is there, but mis-ordered? fix now.
        elif set(meta['Sample_ID']) == set(data_df.columns):
            LOGGER.info(f"Transposed data and reordered meta_data so sample ordering matches.")
            data_df = data_df.transpose() # samples should be in index
            # faster to realign the meta_data instead of the probe data
            sample_order = {v:k for k,v in list(enumerate(data_df.index))}
            # add a temporary column for sorting
            meta['__temp_sorter__'] = meta['Sample_ID'].map(sample_order)
            meta.sort_values('__temp_sorter__', inplace=True)
            meta.drop('__temp_sorter__', axis=1, inplace=True)
            check = True
        elif set(meta['Sample_ID']) == set(data_df.index):
            LOGGER.info(f"Reordered sample meta_data to match data.")
            sample_order = {v:k for k,v in list(enumerate(data_df.index))}
            meta['__temp_sorter__'] = meta['Sample_ID'].map(sample_order)
            meta.sort_values('__temp_sorter__', inplace=True)
            meta.drop('__temp_sorter__', axis=1, inplace=True)
            check = True
    else:
        LOGGER.info('Could not check whether samples in data align with meta_data "Sample_ID" column.')
    if check == False:
        LOGGER.warning("Data samples don't align with 'Sample_ID' column in meta_data.")
    else:
        LOGGER.info("meta.Sample_IDs match data.index (OK)")
    return data_df, meta


def load_both_v1(filepath='.', format='beta_values', file_stem='', verbose=False, silent=False):
    """Loads any pickled files in the given filepath that match specified format,
    plus the associated meta data frame. Returns TWO objects (data, meta) as dataframes for analysis.

    If meta_data files are found in multiple folders, it will read them all and try to match to the samples
    in the beta_values pickles by sample ID.

Arguments:
    filepath:
        Where to look for all the pickle files of processed data.

    format:
        'beta_values', 'm_value', or some other custom file pattern.

    file_stem (string):
        By default, methylprep process with batch_size creates a bunch of generically named files, such as
        'beta_values_1.pkl', 'beta_values_2.pkl', 'beta_values_3.pkl', and so on. IF you rename these or provide
        a custom name during processing, provide that name here.
        (i.e. if your pickle file is called 'GSE150999_beta_values_X.pkl', then your file_stem is 'GSE150999_')

    verbose:
        outputs more processing messages.
    silent:
        suppresses all processing messages, even warnings.
    """
    meta_files = list(Path(filepath).rglob(f'*_meta_data.pkl'))
    multiple_metas = False
    if len(meta_files) > 1:
        LOGGER.info(f"Found several meta_data files; attempting to match each with its respective beta_values files in same folders.")
        multiple_metas = True # this will skip the df-size-match check below.
        ### if multiple_metas, combine them into one dataframe of meta data with
        ### all samples rows and tags in columns
        # Note: this approach assumes that: (a) the goal is a row-wise concatenation (i.e., axis=0) and
        # (b) all dataframes share the same column names.
        # too SLOW meta = pd.concat(meta_files)
        # FASTER: https://gist.github.com/TariqAHassan/fc77c00efef4897241f49e61ddbede9e
        # [supposedly]
        frames = [pd.read_pickle(pkl) for pkl in meta_files]
        def fast_flatten(pkl_list):
            return list(chain.from_iterable(pkl_list))
        meta_tags = frames[0].columns
        # do all match?
        if set([frame.columns for frame in frames]) != set(meta_tags):
            LOGGER.warning(f'Columns in sample sheet meta data files does not match for these files and cannot be combined: {meta_files}. Only loading the first file.')
            meta = pd.read_pickle(meta_files[0])
        df_dict = dict.fromkeys(meta_tags, [])
        for col in meta_tags:
            extracted = (frame[col] for frame in frames) # generator, saves memory
            # flatten
            df_dict[col] = fast_flatten(extracted)
        meta = pd.DataFrame.from_dict(df_dict)[meta_tags]
        # While this method is not very pretty, it typically is much faster than pd.concat() and yields the exact same result.

    if len(meta_files) == 1:
        meta = pd.read_pickle(meta_files[0])
    else:
        LOGGER.info("No meta_data found.")
        meta = pd.DataFrame()

    data_df = load(filepath=filepath,
        format=format,
        file_stem=file_stem,
        verbose=verbose,
        silent=silent
        )

    ### confirm the Sample_ID in meta matches the columns (or index) in data_df.
    check = False
    if 'Sample_ID' in meta.columns:
        if len(meta['Sample_ID']) == len(data_df.columns) and all(meta['Sample_ID'] == data_df.columns):
            data_df = data_df.transpose() # samples should be in index
            check = True
        elif len(meta['Sample_ID']) == len(data_df.index) and all(meta['Sample_ID'] == data_df.index):
            check = True
        # or maybe the data is there, but mis-ordered? fix now.
        elif set(meta['Sample_ID']) == set(data_df.columns):
            LOGGER.info(f"Transposed data and reordered meta_data so sample ordering matches.")
            data_df = data_df.transpose() # samples should be in index
            # faster to realign the meta_data instead of the probe data
            sample_order = {v:k for k,v in list(enumerate(data_df.index))}
            # add a temporary column for sorting
            meta['__temp_sorter__'] = meta['Sample_ID'].map(sample_order)
            meta.sort_values('__temp_sorter__', inplace=True)
            meta.drop('__temp_sorter__', axis=1, inplace=True)
            check = True
        elif set(meta['Sample_ID']) == set(data_df.index):
            LOGGER.info(f"Reordered sample meta_data to match data.")
            sample_order = {v:k for k,v in list(enumerate(data_df.index))}
            meta['__temp_sorter__'] = meta['Sample_ID'].map(sample_order)
            meta.sort_values('__temp_sorter__', inplace=True)
            meta.drop('__temp_sorter__', axis=1, inplace=True)
            check = True
    else:
        LOGGER.info('Could not check whether samples in data align with meta_data "Sample_ID" column.')
    if check == False:
        LOGGER.warning("Data samples don't align with 'Sample_ID' column in meta_data.")
    else:
        LOGGER.info("meta.Sample_IDs match data.index (OK)")
    return data_df, meta


def load(filepath='.', format='beta_values', file_stem='', verbose=False, silent=False):
    """When methylprep processes large datasets, you use the 'batch_size' option to keep memory and file size
    more manageable. Use the `load` helper function to quickly load and combine all of those parts into a single
    data frame of beta-values or m-values.

    Doing this with pandas is about 8 times slower than using numpy in the intermediate step.

    If no arguments are supplied, it will load all files in current directory that have a 'beta_values_X.pkl' pattern.

Arguments:
    filepath:
        Where to look for all the pickle files of processed data.

    format:
        'beta_values', 'm_value', or some other custom file pattern.

    file_stem (string):
        By default, methylprep process with batch_size creates a bunch of generically named files, such as
        'beta_values_1.pkl', 'beta_values_2.pkl', 'beta_values_3.pkl', and so on. IF you rename these or provide
        a custom name during processing, provide that name here.
        (i.e. if your pickle file is called 'GSE150999_beta_values_X.pkl', then your file_stem is 'GSE150999_')

    verbose:
        outputs more processing messages.
    silent:
        suppresses all processing messages, even warnings.
    """
    total_parts = list(Path(filepath).rglob(f'{file_stem}{format}*.pkl'))
    if total_parts == []:
        if not silent:
            LOGGER.warning(f"No pickled files of type ({format}) found in {filepath} (or sub-folders).")
        return
    start = time.process_time()
    parts = []
    #for i in range(1,total_parts):
    probes = pd.DataFrame().index
    samples = pd.DataFrame().index
    for file in tqdm(total_parts, total=len(total_parts), desc="Files"):
        if verbose:
            print(file)
        df = pd.read_pickle(file)
        if len(probes) == 0:
            if df.shape[0] > df.shape[1]:
                probes = df.index
            else:
                probes = df.columns
            if verbose:
                print(f'Probes: {len(probes)}')
        if df.shape[0] > df.shape[1]:
            samples = samples.append(df.columns)
        else:
            samples = samples.append(df.index)
        npy = df.to_numpy()
        parts.append(npy)
    npy = np.concatenate(parts, axis=1) # 8x faster with npy vs pandas
    # axis=1 -- assume that appending to rows, not columns. Each part has same columns (probes)
    try:
        df = pd.DataFrame(data=npy, index=samples, columns=probes)
    except:
        df = pd.DataFrame(data=npy, columns=samples, index=probes)
    if not silent:
        LOGGER.info(f'loaded data {df.shape} from {len(total_parts)} pickled files ({round(time.process_time() - start,3)}s)')
    return df

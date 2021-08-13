import logging
import matplotlib.pyplot as plt
import matplotlib # color maps
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import chain
#from importlib import resources # py3.7+
from .progress_bar import * # context tqdm

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

pkg_namespace = 'methylize.data'
probe2chr = None
probe2map = None
def load_probe_chr_map():
    """__deprecated__ -- manhattan plot function will detect the array type and load this data when needed.
    runs inside manhattan plot."""
    global probe2chr
    global probe2map
    if probe2chr != None:
        return
    # maps probes to chromosomes for all known probes in major array types.
    probe2chr_path = Path(Path(__file__).parent,'data','probe2chr.csv')
    probe2map = pd.read_csv(probe2chr_path)
    #with resources.path(pkg_namespace, 'probe2chr.pkl') as probe2chr_filepath:
    #    with open(probe2chr_filepath,'rb') as f:
    #        probe2map = pickle.load(f)
        # structure is dataframe with 'CGidentifier, CHR, MAPINFO' columns -- bumphunter uses MAPINFO (chromosome position of probes)
    # dict for volcano plots, with a sortable order for chart
    probe2map['CHR'] = probe2map['CHR'].apply(lambda i: f"CHR-0{i}" if i not in ('X','Y') and type(i) is str and int(i) < 10 else f"CHR-{i}")
    probe2map = probe2map[ ~probe2map.CHR.isna() ] # drops control probes or any not linked to a chromosome
    probe2chr = dict(zip(probe2map.CGidentifier, probe2map.CHR)) # computationally fasted conversion method
    # OLD v0.9.3
    #probes = probe2map[['CGidentifier','CHR']].to_dict('records') #SLOW!!!
    #probe2chr = {probe['CGidentifier']:f"CHR-0{probe['CHR']}" if probe['CHR'] not in ('X','Y') and type(probe['CHR']) is str and int(probe['CHR']) < 10 else f"CHR-{probe['CHR']}" for probe in probes}
    #OLD pre v0.9: probe2chr = {k:f"CH-0{v}" if v not in ('X','Y') and type(v) is str and int(v) < 10 else f"CH-{v}" for k,v in probes.items()}

def create_probe_chr_map(manifest):
    """ convert a manifest.data_frame into a dictionary that maps probe_ids to chromosomes, for manhattan plots. """
    probe2map = manifest.data_frame[['CHR','MAPINFO']]
    probe2map = probe2map[ ~(probe2map['CHR'].isin(['*','M'])) ] # mouse manifest has '*' where should be NA, omitting these.
    probe2map = probe2map[ ~probe2map.CHR.isna() ] # drops control probes or any not linked to a chromosome
    probe2map['CHR'] = probe2map['CHR'].copy().apply(lambda i: f"CHR-0{i}" if i not in ('X','Y') and type(i) is str and int(i) < 10 else f"CHR-{i}")
    probe2chr = dict(zip(probe2map.index, probe2map.CHR)) # computationally fasted conversion method
    return probe2chr


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
    """__deprecated__ Maps dataframe to genome locations
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

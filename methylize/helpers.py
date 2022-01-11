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
    probe2map['CHR'] = probe2map['CHR'].apply(lambda i: f"CHR-0{i}" if str(i).isdigit() and int(i) < 10 else f"CHR-{i}")
    probe2map = probe2map[ ~probe2map.CHR.isna() ] # drops control probes or any not linked to a chromosome
    probe2chr = dict(zip(probe2map.CGidentifier, probe2map.CHR)) # computationally fasted conversion method
    # OLD v0.9.3
    #probes = probe2map[['CGidentifier','CHR']].to_dict('records') #SLOW!!!
    #probe2chr = {probe['CGidentifier']:f"CHR-0{probe['CHR']}" if probe['CHR'] not in ('X','Y') and type(probe['CHR']) is str and int(probe['CHR']) < 10 else f"CHR-{probe['CHR']}" for probe in probes}
    #OLD pre v0.9: probe2chr = {k:f"CH-0{v}" if v not in ('X','Y') and type(v) is str and int(v) < 10 else f"CH-{v}" for k,v in probes.items()}

def create_mapinfo(manifest, genome_build=None, include_sex=True):
    """ convert a manifest.data_frame into a dictionary that maps probe_ids to chromosomes, for manhattan plots.
    - genome_build: use NEW or OLD to toggle which genome build to use from manifest.
    - only used by manhattan_plot()"""
    mapinfo = 'OLD_MAPINFO' if genome_build == 'OLD' else 'MAPINFO'
    chr = 'OLD_CHR' if genome_build == 'OLD' else 'CHR'
    sex_chromosomes = ['X','Y']
    probe2map = manifest.data_frame[[mapinfo, chr]]
    probe2map = probe2map[ ~probe2map[mapinfo].isna() ] # drops control probes or any not linked to a chromosome
    if include_sex:
        probe2map = probe2map[ (probe2map[chr].str.isdigit() | probe2map[chr].isin(sex_chromosomes)) ] #manifests have 'X,Y,M,*' omitting these.
    else:
        probe2map = probe2map[ probe2map[chr].str.isdigit() ] #manifests have 'X,Y,M,*' omitting these.
    probe2map[mapinfo] = probe2map[mapinfo].apply(lambda i: f"CHR-0{i}" if str(i).isdigit() and int(i) < 10 else f"CHR-{i}")
    #probe2chr = dict(zip(probe2map.index, probe2map[use])) # computationally fastest conversion method
    return probe2map # dataframe with CHR and MAPINFO columns and probe_names in index.

def create_probe_chr_map(manifest, genome_build=None, include_sex=True):
    """ convert a manifest.data_frame into a dictionary that maps probe_ids to chromosomes, for manhattan plots.
    - genome_build: use NEW or OLD to toggle which genome build to use from manifest.
    - only used by manhattan_plot()"""
    sex_chromosomes = ['X','Y']
    use = 'OLD_CHR' if genome_build == 'OLD' else 'CHR'

    probe2map = manifest.data_frame[[use]]
    probe2map = probe2map[ ~probe2map[use].isna() ] # drops control probes or any not linked to a chromosome
    #other_map = probe2map[ ~probe2map[use].str.isdigit() ]
    if include_sex:
        probe2map = probe2map[ (probe2map[use].str.isdigit() | probe2map[use].isin(sex_chromosomes)) ] #manifests have 'X,Y,M,*' omitting these.
    else:
        probe2map = probe2map[ probe2map[use].str.isdigit() ] #manifests have 'X,Y,M,*' omitting these.
    probe2map[use] = probe2map[use].apply(lambda i: f"CHR-0{i}" if str(i).isdigit() and int(i) < 10 else f"CHR-{i}")
    #probe2map['CHR'] = probe2map['CHR'].copy().apply(lambda i: f"CHR-0{i}" if i not in ('X','Y') and type(i) is str and int(i) < 10 else f"CHR-{i}")
    probe2chr = dict(zip(probe2map.index, probe2map[use])) # computationally fastest conversion method
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
    color_schemes['Gray'] = matplotlib.colors.ListedColormap(['darkgrey','black'])
    color_schemes['Gray2'] = matplotlib.colors.ListedColormap(['darkgrey','gray'])
    color_schemes['Gray3'] = matplotlib.colors.ListedColormap(['whitesmoke','lightgray','silver','darkgray','gray','dimgray','black'])
    color_schemes['Gray4'] = matplotlib.colors.ListedColormap(['slategrey','silver'])
    color_schemes['Volcano'] = matplotlib.colors.ListedColormap(['tab:red','tab:blue','silver'])
    color_schemes['default'] = matplotlib.colors.ListedColormap(['mistyrose', 'navajowhite', 'palegoldenrod', 'yellowgreen', 'mediumseagreen', 'powderblue', 'skyblue',  'lavender', 'plum', 'palevioletred'])
load_color_schemes()


def to_genome(df, rgset): # pragma: no cover
    """__deprecated__ Maps dataframe to genome locations

Parameters:

    df: dataframe
            Dataframe containing methylation, unmethylation, M or Beta
            values for each sample at each site
    rgset: rg channel set instance
            RG channel set instance related to provided df
Returns:

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

def to_BED(stats, manifest_or_array_type, save=True, filename='', genome_build=None, columns=None):
    """Converts & exports manifest and probe p-value dataframe to BED format.
    * https://en.wikipedia.org/wiki/BED_(file_format)
    
    * BED format: [ chromosome number | start position | end position | p-values]
    Where p-values are the output from diff_meth_pos() comparing probes across two or more
    groups of samples for genomic differences in methylation.

    This output is required for combined-pvalues library to read and annotate manhattan plots
    with the nearest Gene(s) for each significant CpG cluster.

    manifest_or_array_type:
        either pass in a Manifest instance from methylprep, or a string that defines which
        manifest to load. One of {'27k', '450k', 'epic', 'epic+', 'mouse'}.
    genome_build:
        pass in 'OLD' to use the older genome build for each respective manifest array type.

    note: if manifest has probes that aren't mapped to genome, they are omitted in BED file.

    TODO: incorporate STRAND and OLD_STRAND in calculations.

    returns a BED formatted dataframe if save is False, or the saved filename if save is True.
    """
    array_types = {'27k', '450k', 'epic', 'epic+', 'mouse'}
    manifest = None
    if isinstance(manifest_or_array_type, str) and manifest_or_array_type not in array_types:
        raise ValueError(f"Specify array type as one of: {array_types}")
    if isinstance(manifest_or_array_type, str) and manifest_or_array_type in array_types:
        import methylprep
        manifest = methylprep.Manifest(methylprep.ArrayType(manifest_or_array_type))
    if not manifest and hasattr(manifest_or_array_type, 'data_frame'):
        manifest = manifest_or_array_type
    if not manifest:
        raise ValueError("Either provide a manifest or specify array_type.")
    if not isinstance(stats, pd.DataFrame):
        raise TypeError("stats should be a dataframe with either a PValue or a FDR_QValue column")
    if not isinstance(manifest.data_frame, pd.DataFrame):
        raise AttributeError("Expected manifest_or_array_type to be a methylprep manifest with a data_frame attribute but this does not have one.")
    if "FDR_QValue" in stats:
        pval = stats['FDR_QValue']
    elif "PValue" in stats:
        pval = stats['PValue']
    else:
        raise IndexError("stats did not contain either a PValue or a FDR_QValue column.")

    # an unfinished, internal undocumented way to change the column names, if exactly 5 columns in list provided in same order.
    if columns is None:
        columns = ['chrom','chromStart','chromEnd','pvalue','name']
        renamer = {}
    else:
        renamer = dict(zip(['chrom','chromStart','chromEnd','pvalue','name'],columns))

    pval = pval.rename("pvalue")
    genes = manifest_gene_map(manifest, genome_build=genome_build)
    # finally, inner join and save/return the combined BED data frame.
    BED = pd.merge(genes[['chrom','chromStart','chromEnd']], pval, left_index=True, right_index=True, how='inner')
    BED = BED.sort_values(['chrom','chromStart'], ascending=True)
    BED = BED.reset_index().rename(columns={'index':'name'})
    BED = BED[['chrom','chromStart','chromEnd','pvalue','name']] # order matters, so be explicit
    # omit unmapped CpGs
    unmapped = len(BED[ BED['chromStart'].isna() ])
    BED = BED[ ~BED['chromStart'].isna() ]
    if renamer != {}:
        BED = BED.rename(columns=renamer)
    # cpv / combined-pvalues needs a tab-separated .bed file
    timestamp = int(time.time())
    if save:
        if isinstance(filename, type(None)):
            BED.to_csv(f"{timestamp}.bed", index=False, sep='\t')
            return f"{timestamp}.bed"
        if not isinstance(filename, Path):
            filename = f"{filename}.bed"
        # otherwise, use as is, assuming it is a complete path/filename
        BED.to_csv(filename, index=False, sep='\t')
        return filename
    return BED

def manifest_gene_map(manifest, genome_build='NEW'):
    """returns 3 columns from manifest for chromosome/gene mapping. Used in >2 functions.
    genome_build: NEW or OLD"""
    ## sort probes by CHR,MAPINFO
    genome_prefix = 'OLD_' if genome_build == 'OLD' else '' # defaults to NEW, no prefix
    if genome_prefix+'MAPINFO' not in manifest.data_frame.columns:
        raise IndexError(f"{genome_prefix+'MAPINFO'} not in manifest")
    genes = manifest.data_frame[[genome_prefix+'CHR', genome_prefix+'MAPINFO']] # Strand and Genome_Build also available, but not needed.
    genes = genes.rename(columns={genome_prefix+'MAPINFO': 'chromStart', genome_prefix+'CHR': 'chrom'}) # standardize, regardless of genome build used
    genes = genes.astype({'chromStart':float})
    genes['chromEnd'] = genes['chromStart'] + 50 # all probes are 50 base pairs long. Ignoring that strand might affect the direction that each probe extends within genome for now. test then fix.
    return genes

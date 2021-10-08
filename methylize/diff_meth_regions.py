import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
#from scipy import stats --- conflicts with input param names
#from joblib import Parallel, delayed, cpu_count
import matplotlib.pyplot as plt
import matplotlib # color maps and changing Agg backend after cpv alters it.
import methylprep
import methylcheck
# from cpv.pipeline import pipeline -- copied and modified here
import numpy as np
from cpv._common import bediter, genomic_control
from pathlib import Path
from .progress_bar import * # tqdm, in_notebook

# cpv imports
import sys
import os
import array
from itertools import groupby, cycle
from operator import itemgetter
#import matplotlib
#matplotlib.use('Agg') --- this prevents plot.show(); only for processing, low overhead
import seaborn as sns
sns.set_context("paper")
sns.set_style("dark", {'axes.linewidth': 1})
from functools import cmp_to_key
try:
    cmp
except NameError:
    def cmp(a, b):
        return (a > b) - (a < b)
import toolshed as ts

# app
from .helpers import color_schemes, create_probe_chr_map, create_mapinfo, to_BED, manifest_gene_map
from .diff_meth_pos import manhattan_plot

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

np.seterr(under='ignore')

__all__ = ['diff_meth_regions', 'pipeline']

def diff_meth_regions(stats, manifest_or_array_type, **kwargs):
    """ wrapper for combined-pvalues pipeline.

about
-----
    comb-p is a command-line tool and a python library that manipulates BED files of possibly irregularly spaced P-values and

    (1) calculates auto-correlation,
    (2) combines adjacent P-values,
    (3) performs false discovery adjustment,
    (4) finds regions of enrichment (i.e. series of adjacent low P-values) and
    (5) assigns significance to those regions.

    ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3496335/

** kwargs : dict
---------------

    stats: dataframe output from diff_meth_pos()
    manifest_or_array_type: class instance or string
    filename: filename to prepend to the .BED file created.
    creates a <filename>.bed output from diff_meth_pos and to_BED.

computational options:
----------------------
dist: int
    maximum distance from each probe to scan for adjacent peaks (80)
acf-dist: int
    window-size for smoothing. Default is 1/3 of peak-dist (dist kwarg),
step: int
    step size for bins in the ACF calculation (50)
threshold: foat
    Extend regions after seeding if pvalue exceeds this value (default: same as seed)
seed: float
    minimum size of a genomic region (0.05)
no_fdr: bool
    don't use FDR
genomic_control: bool
    correct input pvalues for genomic control

display/output options:
-----------------------
verbose: bool -- default False
    Display additional processing information on screen.
p: str
    prefix for output files
region_filter_p:
    max adjusted region-level p-value in final output.
region_filter_n:
    req at least this many probes in a region
annotate:
    label with genes from an external database (e.g. hg19 from cruzdb)
table:
    annotate with refGene

  .. TODO::
  is not "aware" of NEW vs OLD genome builds being used.
    """
    kw = {
        'col_num': 3, # chrom | start | end | pvalue | name
        'step': 50,
        'dist': 80,
        'seed': 0.05,
        'table': 'refGene',
    }
    # user override any defaults
    kw.update(kwargs)

    #TODO use genome_build too
    bed_file = to_BED(stats, manifest_or_array_type, save=True, filename=kwargs.get('filename','stats'), columns=['chrom' ,'start', 'end', 'pvalue', 'name'])

    kw['bed_files'] = [bed_file]
    if not kw.get('prefix'):
        kw['prefix'] = bed_file.split('.')[0]

    if isinstance(manifest_or_array_type, methylprep.Manifest):
        manifest = manifest_or_array_type
    elif isinstance(manifest_or_array_type, str) and manifest_or_array_type in methylprep.files.manifests.ARRAY_FILENAME.keys():
        manifest = methylprep.Manifest(methylprep.ArrayType(manifest_or_array_type))

    try:
        results = pipeline(kw['col_num'], kw['step'], kw['dist'],
            kw.get('acf_dist', int(round(0.33333 * kw['dist'], -1))),
            kw.get('prefix',''), kw.get('threshold', kw['seed']),
            kw['seed'], kw['table'], kw['bed_files'],
            region_filter_p = kw.get('region_filter_p',1),
            region_filter_n = kw.get('region_filter_n'),
            genome_control=False, db=True, use_fdr=not kw.get('no_fdr',False),
            log_to_file=True, verbose=kw.get('verbose',False))
        LOGGER.info(results)
    except SystemExit as e:
        LOGGER.info("No regions found")
        LOGGER.info(e)
    # add probe names back into these files:
    files_created = [
        f"{kw.get('prefix','')}.args.txt",
        f"{kw.get('prefix','')}.acf.txt",
        f"{kw.get('prefix','')}.fdr.bed.gz",
        f"{kw.get('prefix','')}.slk.bed.gz",
        f"{kw.get('prefix','')}.regions.bed.gz",
        f"{kw.get('prefix','')}.regions-p.bed.gz",
    ]
    chromStart = manifest.data_frame['MAPINFO'].astype(float).reset_index() # _OLD or NEW?
    # add probe names and prepare one common stats-results CSV:
    stats_series = {}
    _deleted = None
    _added = None
    for _file in files_created:
        # errors here are for when no regions are found, and file are blank.
        if 'regions.bed' in _file:
            regions_header = ['chrom','chromStart','chromEnd','min_p','n_probes']
            # regions-p includes all of regions; delete below.
            Path(_file).unlink()
            _deleted = _file
        elif 'regions-p.bed' in _file:
            regions_p_header = ['chrom','chromStart','chromEnd','min_p','n_probes','z_p','z_sidak_p']
            regions_p_rename = {'#chrom':'chrom', 'start': 'chromStart', 'end': 'chromEnd'}
            try:
                df = pd.read_csv(_file, sep='\t').rename(columns=regions_p_rename)
                df = df.merge(chromStart, left_on='chromStart', right_on='MAPINFO', how='inner').drop(columns=['MAPINFO']).rename(columns={'IlmnID':'name'})
                regions_stats_file = f"{kw.get('prefix','')}_regions.csv"
                df.to_csv(regions_stats_file, index=False)
                _added = regions_stats_file
            except Exception as e:
                LOGGER.error(f"{_file}: {e}")
        elif '.bed' in Path(_file).suffixes:
            try:
                df = pd.read_csv(_file, sep='\t')
                # match using start position (MAPINFO)
                df = df.merge(chromStart, left_on='start', right_on='MAPINFO', how='inner').drop(columns=['MAPINFO']).rename(columns={'IlmnID':'name'})
                df.to_csv(_file, sep='\t', index=False)
                if 'fdr' in _file:
                    stats_series[_file] = df[['p', 'region-p','region-q','name']].set_index('name').rename(
                        columns={'p': 'fdr-p', 'region-p': 'region-p', 'region-q': 'fdr-region-q'}) # might PValue be FDR_QValue instead?
                if 'slk' in _file:
                    stats_series[_file] = df[['region-p','name']].set_index('name').rename(
                        columns={'region-p': 'slk-region-p'})
            except Exception as e:
                LOGGER.error(f"{_file}: {e}")
    files_created.remove(_deleted)
    files_created.append(_added)
    # switch from `Agg` to interactive; but diff OSes work with diff ones, and the best ones are not default installed.
    interactive_backends = ['Qt5Agg', 'MacOSX', 'TkAgg', 'ipympl', 'GTK3Agg', 'GTK3Cairo', 'nbAgg', 'Qt5Cairo','TkCairo']
    if in_notebook():
        try:
            matplotlib.use('ipympl')
        except:
            pass
    for backend in interactive_backends:
        try:
            matplotlib.use(backend)
        except:
            continue

    manhattan_cols = {'region-p':'PValue', '#chrom':'chromosome', 'start': 'MAPINFO'}
    _fdr_ = pd.read_csv(kw['prefix'] + '.fdr.bed.gz', sep='\t').rename(columns=manhattan_cols).set_index('name')
    manhattan_plot(_fdr_, manifest)

    if stats_series != {}:
        # problem: manifest is unique but FDR/SLK, so need to look up
        chr_start_end = manifest_gene_map(manifest, genome_build='NEW')
        chr_start_end.index.name = 'name'
        probe_index = list(stats_series.values())[0].index
        #--- cannot DO: stats_series['chrom'] = chr_start_end[ chr_start_end.index.isin(probe_index) ]
        # _file = f"{kw.get('prefix','')}.fdr.bed.gz"
        # concat fails because FDR/SLK probes can repeat in index. must merge first.
        stats_df = pd.concat(list(stats_series.values()), axis=1)
        try:
            stats_df = stats_df.merge(chr_start_end, left_index=True, right_index=True)
        except:
            LOGGER.error('Could not includes chrom | chromStart | chromEnd in stats file.')
        stats_file = f"{kw.get('prefix','')}_stats.csv"
        stats_df.to_csv(stats_file)
        files_created.append(stats_file)
    return files_created

def qqplot(lpys, ax_qq):
    lunif = -np.log10(np.arange(1, len(lpys) + 1) / float(len(lpys)))[::-1]
    ax_qq.plot(lunif, np.sort(lpys), marker=',', linestyle='none', c='#EA352B')
    ax_qq.set_xticks([])
    ax_qq.plot(lunif, lunif, ls='--', c='#959899')
    ax_qq.set_xlabel('')
    ax_qq.set_ylabel('')
    ax_qq.set_yticks([])
    ax_qq.axis('tight')
    ax_qq.axes.set_frame_on(True)

def read_regions(fregions):
    """Reads a BED file (tab separated CSV with chrom, start, end) and returns a lookup
    dict of chromosomes and their cpg regions"""
    if not fregions: return None
    df = pd.read_csv(fregions, sep='\t').rename(columns={'#chrom': 'chrom'})
    if set(df.columns) & set(['chrom','chromStart','chromEnd']) != set(['chrom','chromStart','chromEnd']):
        raise KeyError(f"BED columns should include ['chrom','chromStart','chromEnd']; found {set(df.columns)}")
    regions = {k:[] for k in df['chrom'].unique()} # split by chromosome into a list of (start,end) tuples
    #for idx,row in df.iterrows():
    #    regions[row['chrom']].append((int(row['chromStart']), int(row['chromEnd'])))
    chromosomes = df.groupby('chrom')
    for chromosome_number, chromosome in chromosomes:
        pairs = list(chromosome.apply(lambda x: (x['chromStart'], x['chromEnd']), axis=1) )
        regions[chromosome_number] = pairs
    return regions



def pipeline(col_num, step, dist, acf_dist, prefix, threshold, seed, table,
        bed_files, mlog=True, region_filter_p=1, region_filter_n=None,
        genome_control=False, db=None, use_fdr=True, log_to_file=True, verbose=False):
    # a hack to ensure local files can be imported
    # sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from cpv import acf, slk, fdr, peaks, region_p, stepsize, filter
    from cpv._common import genome_control_adjust, genomic_control, bediter
    import operator


    if step is None:
        step = min(acf_dist, stepsize.stepsize(bed_files, col_num))
        if verbose: LOGGER.info(f"calculated stepsize as: {step}")

    lags = list(range(1, acf_dist, step))
    lags.append(lags[-1] + step)

    prefix = prefix.rstrip(".")

    # ACF: auto-correlation
    putative_acf_vals = acf.acf(bed_files, lags, col_num, simple=False,
                                mlog=mlog)
    acf_vals = []
    # go out to max requested distance but stop once an autocorrelation
    # < 0.05 is added.
    for a in putative_acf_vals:
        # a is ((lmin, lmax), (corr, N))
        # this heuristic seems to work. stop just above the 0.08 correlation
        # lag.
        if a[1][0] < 0.04 and len(acf_vals) > 2: break
        acf_vals.append(a)
        if a[1][0] < 0.04 and len(acf_vals): break

    if log_to_file:
        # save the arguments that this was called with.
        with open(prefix + ".args.txt", "w") as fh:
            sys_args = " ".join(sys.argv[1:])
            print(sys_args + "\n", file=fh) #### <<<--- only catch command line args
            import datetime
            print("date: %s" % datetime.datetime.today(), file=fh)
            from .__init__ import __version__
            print("version:", __version__, file=fh)
            if verbose: LOGGER.info(f"{sys_args} | {timestamp} | {__version__}")

        with open(prefix + ".acf.txt", "w") as fh:
            acf_vals = acf.write_acf(acf_vals, fh)
            print("wrote: %s" % fh.name, file=fh)
        if verbose: LOGGER.info(f"ACF: {acf_vals}")
    else:
        if verbose: LOGGER.info(f"ACF: {acf_vals}")

    spvals, opvals = array.array('f'), array.array('f')
    with ts.nopen(prefix + ".slk.bed.gz", "w") as fhslk:
        fhslk.write('#chrom\tstart\tend\tp\tregion-p\n')
        for chrom, results in slk.adjust_pvals(bed_files, col_num, acf_vals):
            fmt = chrom + "\t%i\t%i\t%.4g\t%.4g\n"
            for row in results:
                row = tuple(row)
                fhslk.write(fmt % row)
                opvals.append(row[-2])
                spvals.append(row[-1])

    if verbose: LOGGER.info(f"Original lambda: {genomic_control(opvals)}")
    del opvals

    gc_lambda = genomic_control(spvals)
    if verbose: LOGGER.info(f"wrote: {fhslk.name} with lambda: {gc_lambda}")

    if genome_control:
        fhslk = ts.nopen(prefix + ".slk.gc.bed.gz", "w")
        adj = genome_control_adjust([d['p'] for d in bediter(prefix + ".slk.bed.gz", -1)])
        for i, line in enumerate(ts.nopen(prefix + ".slk.bed.gz")):
            print("%s\t%.5g" % (line.rstrip("\r\n"), adj[i]), file=fhslk)

        fhslk.close()
        if verbose: LOGGER.info(f"wrote: {fhslk.name}")

    with ts.nopen(prefix + ".fdr.bed.gz", "w") as fh:
        fh.write('#chrom\tstart\tend\tp\tregion-p\tregion-q\n')
        for bh, l in fdr.fdr(fhslk.name, -1):
            fh.write("%s\t%.4g\n" % (l.rstrip("\r\n"), bh))
        if verbose: LOGGER.info(f"wrote: {fh.name}")
    fregions = prefix + ".regions.bed.gz"
    with ts.nopen(fregions, "w") as fh:
        list(peaks.peaks(prefix + ".fdr.bed.gz", -1 if use_fdr else -2, threshold, seed,
            dist, fh, operator.le))
    n_regions = sum(1 for _ in ts.nopen(fregions))
    if verbose: LOGGER.info(f"wrote: {fregions} ({n_regions} regions)")
    if n_regions == 0:
        LOGGER.warning("No regions found.")
        return

    # HACK -- edit pvalues and region-p to be >0.00000
    # this prevents a bunch of "divide by zero" warnings
    temp = pd.read_csv(prefix + ".slk.bed.gz", sep='\t')
    temp['p'] = temp['p'].apply(lambda x: 0.0000001 if x == 0 else x)
    temp['region-p'] = temp['region-p'].apply(lambda x: 0.0000001 if x == 0 else x)
    temp.to_csv(prefix + ".slk.bed.gz", sep='\t', index=False)

    with ts.nopen(prefix + ".regions-p.bed.gz", "w") as fh:
        N = 0
        fh.write("#chrom\tstart\tend\tmin_p\tn_probes\tz_p\tz_sidak_p\n")
        # use -2 for original, uncorrected p-values in slk.bed
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for region_line, slk_p, slk_sidak_p, sim_p in region_p.region_p(
                                   prefix + ".slk.bed.gz",
                                   prefix + ".regions.bed.gz", -2,
                                   step):
                fh.write("%s\t%.4g\t%.4g\n" % (region_line, slk_p, slk_sidak_p))
                fh.flush()
                N += int(slk_sidak_p < 0.05)
            LOGGER.info(f"wrote: {fh.name}, (regions with corrected-p < 0.05: {N})")

    regions_bed = fh.name
    #if all(h in header for h in ('t', 'start', 'end')):
    if region_filter_n is None:
        region_filter_n = 0

    """ NEXT function filter.filter() requires bedtools installed, and only works on macos/linux.
    with ts.nopen(prefix + ".regions-t.bed", "w") as fh:
        N = 0
        for i, toks in enumerate(filter.filter(bed_files[0], regions_bed, p_col_name=col_num)):
            if i == 0: toks[0] = "#" + toks[0]
            else:
                if float(toks[6]) > region_filter_p: continue
                if int(toks[4]) < region_filter_n: continue
                #if region_ and "/" in toks[7]:
                #    # t-pos/t-neg. if the lower one is > region_?
                #    vals = map(int, toks[7].split("/"))
                #    if min(vals) > region_: continue

                N += 1
            print("\t".join(toks), file=fh)
        print(("wrote: %s, (regions with region-p "
                            "< %.3f and n-probes >= %i: %i)") \
                % (fh.name, region_filter_p, region_filter_n, N),
                file=sys.stderr)
    """
    from cpv import manhattan
    regions = manhattan.read_regions(fh.name)

    manhattan.manhattan(prefix + ".slk.bed.gz", 3,
        prefix.rstrip(".") + ".manhattan.png",
        False, ['#959899', '#484B4C'], "", False, None,
        regions=regions, bonferonni=False)

    """ cruzdb is python2x only, and hasn't been maintained since 2014.
    if db is not None:
        from cruzdb import Genome
        g = Genome(db)
        lastf = fh.name
        with open(prefix + ".anno.%s.bed" % db, "w") as fh:
            fh.write('#')
            g.annotate(lastf, (table, "cpgIslandExt"), out=fh,
                    feature_strand=True, parallel=len(spvals) > 500)
        print("wrote: %s annotated with %s %s" % (fh.name, db, table), file=sys.stderr)
    """
    return

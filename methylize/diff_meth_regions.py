import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
#from scipy import stats --- conflicts with input param names
#from joblib import Parallel, delayed, cpu_count
import matplotlib # color maps and changing Agg backend after cpv alters it.
default_backend = matplotlib.get_backend()
import matplotlib.pyplot as plt
import methylprep
import numpy as np
# from cpv.pipeline import pipeline -- copied and modified here
#from methylize.cpv._common import bediter, genomic_control
import methylize
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
from .helpers import color_schemes, to_BED, manifest_gene_map
from .diff_meth_pos import manhattan_plot
from .genome_browser import fetch_genes

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

np.seterr(under='ignore')

__all__ = ['diff_meth_regions']

def diff_meth_regions(stats, manifest_or_array_type, **kwargs):
    """Calculates and annotates diffentially methylated regions (DMR) using the `combined-pvalues pipeline` and returns list of output files.

comb-p is a command-line tool and a python library that manipulates BED files of possibly irregularly spaced P-values and

(1) calculates auto-correlation,
(2) combines adjacent P-values,
(3) performs false discovery adjustment,
(4) finds regions of enrichment (i.e. series of adjacent low P-values) and
(5) assigns significance to those regions.

ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3496335/

Input Parameters:

    stats: dataframe
        dataframe output from diff_meth_pos()
    manifest_or_array_type: class instance or string
        pass in the manifest, or the name of the array
    filename:
        filename to prepend to the .BED file created.
    creates a <filename>.bed output from diff_meth_pos and to_BED.
    genome_build: default 'NEW'
        by default, it uses the NEWer genome build. Each manifest contains two genome builds,
        marked "NEW" and "OLD". To use the OLD build, se this to "OLD".

Computational Parameters:

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
    genome_control: bool
        correct input pvalues for genome control (inflation factor). This reduces the confounding effects
        of population stratitication in EWAS data.

Display/output Paramters:

    verbose: bool -- default False
        Display additional processing information on screen.
    plot: bool -- default False
        False will suppress the manhattan plot step.
    prefix: str
        prefix that gets appended to all output files (e.g. 'dmr' becomes 'dmr_regions.csv')
    region_filter_p:
        max adjusted region-level p-value in final output.
    region_filter_n:
        req at least this many probes in a region
    annotate: bool
        annotate with fetch_genes() function that uses UCSC refGene database to add "nearby" genes to
        differentially methylated regions in the output CSV. If you want to fine-tune the reference database,
        the tolerance of what "nearby" means, and other parameters, set this to false and call `methylize.fetch_genes`
        as a separate step on the '..._regions.csv' output file.
    tissue: str
        if specified, adds additional columns to the annotation output with the expression levels for identified genes
        in any/all tissue(s) that match the keyword. (e.g. if your methylation samples are whole blood,
        specify `tissue=blood`) For all 54 tissues, use `tissue=all`

Returns:

    list
        A list of files created.
    """
    kw = {
        'col_num': 3, # chrom | start | end | pvalue | name
        'step': None, # comb-p default is 50, but this will auto-calc
        'dist': 80,
        'seed': 0.05,
        'table': 'refGene',
    }
    # user override any defaults
    kw.update(kwargs)

    bed_filename = f"{kwargs.get('prefix','')}_dmp_stats"
    bed_file = to_BED(stats, manifest_or_array_type,
        save=True,
        filename=bed_filename,
        columns=['chrom' ,'start', 'end', 'pvalue', 'name'],
        genome_build=kw.get('genome_build', None)
        )

    kw['bed_files'] = [bed_file]
    if not kw.get('prefix'):
        kw['prefix'] = bed_file.split('.')[0]

    if isinstance(manifest_or_array_type, methylprep.Manifest):
        manifest = manifest_or_array_type
    elif isinstance(manifest_or_array_type, str) and manifest_or_array_type in methylprep.files.manifests.ARRAY_FILENAME.keys():
        manifest = methylprep.Manifest(methylprep.ArrayType(manifest_or_array_type))

    try:
        results = _pipeline(kw['col_num'], kw['step'], kw['dist'],
            kw.get('acf_dist', int(round(0.33333 * kw['dist'], -1))),
            kw.get('prefix',''), kw.get('threshold', kw['seed']),
            kw['seed'], kw['table'], kw['bed_files'],
            region_filter_p = kw.get('region_filter_p',1),
            region_filter_n = kw.get('region_filter_n'),
            genome_control=kw.get('genome_control', False), use_fdr=not kw.get('no_fdr',False),
            log_to_file=True, verbose=kw.get('verbose',False))
        if results["result"] != "OK":
            LOGGER.warning(results)
    except Exception as e:
        import traceback
        LOGGER.error(traceback.format_exc())
        LOGGER.error(e)
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
    _no_regions = None
    _added = None
    for _file in files_created:
        # errors here are for when no regions are found, and file are blank.
        if 'regions.bed' in _file:
            regions_header = ['chrom','chromStart','chromEnd','min_p','n_probes']
            # regions-p includes all of regions; delete below.
            Path(_file).unlink()
            _deleted = _file
        elif 'regions-p.bed' in _file:
            if results["result"] == "No regions found":
                _no_regions = _file
                continue
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
    if _no_regions:
        files_created.remove(_no_regions)
    # switch from `Agg` to interactive; but diff OSes work with diff ones, and the best ones are not default installed.
    interactive_backends = ['TkAgg', 'Qt5Agg', 'MacOSX', 'ipympl', 'GTK3Agg', 'GTK3Cairo', 'nbAgg', 'Qt5Cairo','TkCairo']
    try:
        matplotlib.switch_backend(default_backend)
    except:
        if in_notebook():
            try:
                matplotlib.switch_backend('ipympl')
            except:
                pass
        for backend in interactive_backends:
            try:
                matplotlib.switch_backend(backend)
            except:
                continue

    if kw.get('plot') == True:
        try:
            manhattan_cols = {'region-p':'PValue', 'region-q':'FDR_QValue', '#chrom':'chromosome', 'start': 'MAPINFO'}
            _fdr_ = pd.read_csv(kw['prefix'] + '.fdr.bed.gz', sep='\t').rename(columns=manhattan_cols).set_index('name')
            manhattan_plot(_fdr_, manifest)
        except Exception as e:
            if kw.get('verbose',False) == True:
                LOGGER.error("Could not produce the manhattan plot: {e}")

    if stats_series != {}:
        # problem: manifest is unique but FDR/SLK, so need to look up
        chr_start_end = manifest_gene_map(manifest, genome_build=kw.get('genome_build', None))
        chr_start_end.index.name = 'name'
        probe_index = list(stats_series.values())[0].index
        #--- cannot DO: stats_series['chrom'] = chr_start_end[ chr_start_end.index.isin(probe_index) ]
        # _file = f"{kw.get('prefix','')}.fdr.bed.gz"
        # concat fails because FDR/SLK probes can repeat in index. must merge first.
        try:
            stats_df = pd.concat(list(stats_series.values()), axis=1)
            stats_df = stats_df.merge(chr_start_end, left_index=True, right_index=True)
            stats_file = f"{kw.get('prefix','')}_stats.csv"
            stats_df.to_csv(stats_file)
            files_created.append(stats_file)
        except Exception as e:
            LOGGER.error(f'Could not include chrom | chromStart | chromEnd in stats file: {e} (some probes appear in multiple results rows)')

    # cruzdb is python2x only, and hasn't been maintained since 2014. So we wrote our own UCSC interface function: fetch_genes
    regions_stats_file = Path(f"{kw.get('prefix','')}_regions.csv")
    if kw.get('annotate',True) == True and regions_stats_file.exists():
        if manifest.array_type in ['mouse']:
            LOGGER.warning(f"Genome annotation is not supported for {manifest.array_type} array_type.")
        elif kw.get('genome_build', None) == 'OLD':
            LOGGER.warning(f"Genome annotation is not supported for OLD genome builds. Only the latest (hg38) build is supported.")
        else:
            final_results = fetch_genes(regions_stats_file, tissue=kw.get('tissue',None))
            files_created.append(f"{kw.get('prefix','')}_regions_genes.csv")
    elif kw.get('annotate',True) == True:
        LOGGER.error(f"Could not annotate; no regions.csv file found")
    files_created.append(bed_file)
    return files_created


def _pipeline(col_num0, step, dist, acf_dist, prefix, threshold, seed, table,
        bed_files, mlog=True, region_filter_p=1, region_filter_n=None,
        genome_control=False, use_fdr=True, log_to_file=True, verbose=False):
    """Internal pipeline: adapted from `combined-pvalues` (cpv) pipeline to work outside of a CLI."""
    # a hack to ensure local files can be imported
    # sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    #from cpv import acf, slk, fdr, peaks, region_p, stepsize, filter
    #from cpv._common import genome_control_adjust, genomic_control, bediter
    import operator


    if step is None:
        step = min(acf_dist, methylize.cpv.stepsize(bed_files, col_num0))
        if verbose: LOGGER.info(f"calculated stepsize as: {step}")

    lags = list(range(1, acf_dist, step))
    lags.append(lags[-1] + step)

    prefix = prefix.rstrip(".")

    # ACF: auto-correlation
    putative_acf_vals = methylize.cpv.acf(bed_files, lags, col_num0, simple=False,
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
            timestamp = datetime.datetime.today()
            print("date: %s" % timestamp, file=fh)
            from .__init__ import __version__
            print("version:", __version__, file=fh)
            if verbose: LOGGER.info(f"{sys_args} | {timestamp} | {__version__}")

        with open(prefix + ".acf.txt", "w") as fh:
            acf_vals = methylize.cpv.write_acf(acf_vals, fh)
            print("wrote: %s" % fh.name, file=fh)
        if verbose: LOGGER.info(f"ACF: {acf_vals}")
    else:
        if verbose: LOGGER.info(f"ACF: {acf_vals}")

    spvals, opvals = array.array('f'), array.array('f')
    with ts.nopen(prefix + ".slk.bed.gz", "w") as fhslk:
        fhslk.write('#chrom\tstart\tend\tp\tregion-p\n')
        for chrom, results in methylize.cpv.slk.adjust_pvals(bed_files, col_num0, acf_vals):
            fmt = chrom + "\t%i\t%i\t%.4g\t%.4g\n"
            for row in results:
                row = tuple(row)
                fhslk.write(fmt % row)
                opvals.append(row[-2])
                spvals.append(row[-1])

    if verbose: LOGGER.info(f"Original lambda (genomic control): {methylize.cpv.genomic_control(opvals)}")
    del opvals

    gc_lambda = methylize.cpv.genomic_control(spvals)
    if verbose: LOGGER.info(f"wrote: {fhslk.name} with lambda: {gc_lambda}")

    if genome_control:
        # adjust p-values by the genomic inflance control factor, lambda
        # see https://en.wikipedia.org/wiki/Genomic_control
        # or https://onlinelibrary.wiley.com/doi/abs/10.1111/j.0006-341X.1999.00997.x for explanation
        adj = methylize.cpv.genome_control_adjust([d['p'] for d in methylize.cpv.bediter(prefix + ".slk.bed.gz", -1)])
        slk_df = pd.read_csv(prefix + ".slk.bed.gz", sep='\t')
        slk_df['original_p'] = slk_df['p']
        slk_df['p'] = adj
        slk_df.to_csv(prefix + ".slk.bed.gz", sep='\t', index=False)

    with ts.nopen(prefix + ".fdr.bed.gz", "w") as fh:
        fh.write('#chrom\tstart\tend\tp\tregion-p\tregion-q\n')
        for bh, l in methylize.cpv.fdr(fhslk.name, -1):
            fh.write("%s\t%.4g\n" % (l.rstrip("\r\n"), bh))
        if verbose: LOGGER.info(f"wrote: {fh.name}")
    fregions = prefix + ".regions.bed.gz"
    with ts.nopen(fregions, "w") as fh:
        list(methylize.cpv.peaks(prefix + ".fdr.bed.gz", -1 if use_fdr else -2, threshold, seed,
            dist, fh, operator.le))
    n_regions = sum(1 for _ in ts.nopen(fregions))
    if verbose: LOGGER.info(f"wrote: {fregions} ({n_regions} regions)")
    if n_regions == 0:
        return {"result": "No regions found"}

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
            for region_line, slk_p, slk_sidak_p, sim_p in methylize.cpv.region_p(
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
    if not Path(f"{prefix}.regions-p.bed.gz").exists():
        return {"result": "No clustered CpG regions found (that differ between your sample groups)."}

    """combined-pvalues: NEXT function filter.filter() requires bedtools installed, and only works on macos/linux.
    -- combines regions from multiple datasets, I think.
    with ts.nopen(prefix + ".regions-t.bed", "w") as fh:
        N = 0
        for i, toks in enumerate(filter.filter(bed_files[0], regions_bed, p_col_name=col_num0)):
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
    #if verbose:
    #    regions = methylize.cpv.read_regions(fh.name)
    #    methylize.cpv.manhattan(prefix + ".slk.bed.gz", 3,
    #        prefix.rstrip(".") + ".manhattan.png",
    #        False, ['#959899', '#484B4C'], "", False, None,
    #        regions=regions, bonferonni=False)
    return {"result": "OK"}



""" NOT USED
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
"""

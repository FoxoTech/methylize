from cpv import slk, acf
# Here is all of the necessary code from comb-p package to run DMR.
# I tried to import the package and use it, but everything runs via command line and uses excessive disk storage
# for I/O, which was causing lots of errors, so consolidating it to be bug free with simpler GUI in notebooks.
""" pipeline imports """
import sys
import signal
import os.path as op
import toolshed as ts # nopen reader header
import operator
import threading
from itertools import tee
# python 2-->3 vars
izip = zip
basestring = str
long = int
"""
   calculate the step-size that should be used for the ACF calculations.
   The step-size is calculated as::
       median(distance-between-adjacent-starts)
   This heuristic seems to work well for creating bins with equal amounts of
   records for the ACF.
"""
import argparse
#from _common import get_col_num, bediter
from operator import itemgetter
from itertools import groupby
import numpy as np
"""
   calculate the autocorrelation of a *sorted* bed file with a set
   of *distance* lags.
"""
import argparse
from array import array
import sys
import numpy as np
import scipy.stats as ss
try:
    from itertools import groupby, izip, chain
except ImportError:
    from itertools import groupby, chain
    izip = zip
    xrange = range
from cpv._common import bediter, pairwise, get_col_num, get_map

"""
find peaks or troughs in sorted bed files
for a bedgraph file with pvalues in the 4th column. usage would be:
    $ python peaks.py --dist 100 --seed 0.01 some.bed > some.regions.bed
where some.regions.bed contains the start and end of the region and (currently)
the lowest p-value in that region.
"""
from itertools import groupby
import operator
from toolshed import reader
import argparse
import sys
"""
   calculate a p-value of a region using the Stouffer-Liptak method or the
   z-score method.
"""
import argparse
import sys
import numpy as np
import toolshed as ts
from collections import defaultdict
from interlap import InterLap
from itertools import chain, groupby, combinations
from operator import itemgetter
#from stouffer_liptak import stouffer_liptak, z_score_combine

from scipy.stats import norm
from numpy.linalg import cholesky as chol
qnorm = norm.ppf
pnorm = norm.cdf
"""
count the number of switches in sign in the regions. Since the region
calculation is based on the p-value only, it could be that a region is
discovered that has both high and low t-scores.
This script will output the original region_bed intervals, along with
sum of positive t-scores and the sum of negative t-scores.
"""
import argparse
from operator import itemgetter
from itertools import groupby
from tempfile import mktemp
try:
    long
except NameError:
    long = int
from math import exp
import atexit
import os

"""
    from stouffer-lipgloss.py
"""
import sys
import numpy as np
from scipy.stats import norm
import scipy.stats as ss
from numpy.linalg import cholesky as chol
from numpy.linalg.linalg import LinAlgError
qnorm = norm.ppf
pnorm = norm.cdf

chisqprob = ss.distributions.chi2.sf

import toolshed as ts

import logging
LOGGER = logging.getLogger(__name__)

def pipeline(col_num=3, step=None, dist=80, acf_dist=None, prefix='dmr',
             threshold=None, seed=0.05, table='refGene',
             bed_files=[], mlog=True, region_filter_p=1, region_filter_n=None,
             genome_control=False, db=None, use_fdr=True):
    """ step None: auto detect
    bed_files must be filled in.
    """
    if not acf_dist:
        acf_dist = int(round(0.33333 * dist, -1))
    if not threshold:
        threshold = seed

    #sys.path.insert(0, op.join(op.dirname(__file__), ".."))

    # acf, fdr (without qvality)
    # slk, peaks, region_p, stepsize, filter
    # from cpv._common import genome_control_adjust, genomic_control, bediter


    if step is None:
        step = min(acf_dist, stepsize(bed_files, col_num))
        print("calculated stepsize as: %i" % step, file=sys.stderr)

    lags = list(range(1, acf_dist, step))
    lags.append(lags[-1] + step)

    prefix = prefix.rstrip(".")

    from cpv import acf
    putative_acf_vals = acf.acf(bed_files, lags, col_num, simple=False,
                                partial=True, mlog=mlog)
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

    # save the arguments that this was called with.
    """
    with open(prefix + ".args.txt", "w") as fh:
        print(" ".join(sys.argv[1:]) + "\n", file=fh)
        import datetime
        print("date: %s" % datetime.datetime.today(), file=fh)
        from .__init__ import __version__
        print("version:", __version__, file=fh)

    with open(prefix + ".acf.txt", "w") as fh:
        acf_vals = acf.write_acf(acf_vals, fh)
        print("wrote: %s" % fh.name, file=fh)
    """
    LOGGER.info(f"ACF: {acf_vals}")

    spvals, opvals = array('f'), array('f')

    with ts.nopen(prefix + ".slk.bed.gz", "w") as fhslk:
        fhslk.write('#chrom\tstart\tend\tp\tregion-p\n')
        for chrom, results in slk.adjust_pvals(bed_files, col_num, acf_vals):
            print(chrom)
            fmt = chrom + "\t%i\t%i\t%.4g\t%.4g\n"
            for row in results:
                row = tuple(row)
                fhslk.write(fmt % row)
                opvals.append(row[-2])
                spvals.append(row[-1])

    print("# original lambda: %.2f" % genomic_control(opvals), file=sys.stderr)
    del opvals

    gc_lambda = genomic_control(spvals)
    #print("wrote: %s with lambda: %.2f" % (fhslk.name, gc_lambda), file=sys.stderr)

    if genome_control:
        LOGGER.info('genomic control lambda adjust')
        fhslk = ts.nopen(prefix + ".slk.gc.bed.gz", "w")
        adj = genome_control_adjust([d['p'] for d in bediter(prefix + ".slk.bed.gz", -1)])
        for i, line in enumerate(ts.nopen(prefix + ".slk.bed.gz")):
            print("%s\t%.5g" % (line.rstrip("\r\n"), adj[i]), file=fhslk)

        #fhslk.close()
        #print("wrote: %s" % fhslk.name, file=sys.stderr)

    # load with pandas instead of toolshed.

    #####
    """
    with ts.nopen(prefix + ".fdr.bed.gz", "w") as fh:
        fh.write('#chrom\tstart\tend\tp\tregion-p\tregion-q\n')
        for bh, l in fdr.fdr(fhslk.name, -1):
            fh.write("%s\t%.4g\n" % (l.rstrip("\r\n"), bh))
        print("wrote: %s" % fh.name, file=sys.stderr)
    """
    fregions = prefix + ".regions.bed.gz"
    with ts.nopen(fregions, "w") as fh:
        list(peaks.peaks(prefix + ".fdr.bed.gz", -1 if use_fdr else -2, threshold, seed,
            dist, fh, operator.le))
    n_regions = sum(1 for _ in ts.nopen(fregions))
    print("wrote: %s (%i regions)" % (fregions, n_regions), file=sys.stderr)
    if n_regions == 0:
        sys.exit()

    with ts.nopen(prefix + ".regions-p.bed.gz", "w") as fh:
        N = 0
        fh.write("#chrom\tstart\tend\tmin_p\tn_probes\tz_p\tz_sidak_p\n")
        # use -2 for original, uncorrected p-values in slk.bed
        for region_line, slk_p, slk_sidak_p, sim_p in region_p.region_p(
                               prefix + ".slk.bed.gz",
                               prefix + ".regions.bed.gz", -2,
                               step):
            fh.write("%s\t%.4g\t%.4g\n" % (region_line, slk_p, slk_sidak_p))
            fh.flush()
            N += int(slk_sidak_p < 0.05)
        print("wrote: %s, (regions with corrected-p < 0.05: %i)" \
                % (fh.name, N), file=sys.stderr)

    regions_bed = fh.name
    #if all(h in header for h in ('t', 'start', 'end')):
    if region_filter_n is None: region_filter_n = 0
    with ts.nopen(prefix + ".regions-t.bed", "w") as fh:
        N = 0
        for i, toks in enumerate(bed_filter(bed_files[0],
            regions_bed, p_col_name=col_num)):
            if i == 0: toks[0] = "#" + toks[0]
            else:
                if float(toks[6]) > region_filter_p: continue
                if int(toks[4]) < region_filter_n: continue
                #if region_filter_t and "/" in toks[7]:
                #    # t-pos/t-neg. if the lower one is > region_filter_t?
                #    vals = map(int, toks[7].split("/"))
                #    if min(vals) > region_filter_t: continue

                N += 1
            print("\t".join(toks), file=fh)
        print(("wrote: %s, (regions with region-p "
                            "< %.3f and n-probes >= %i: %i)") \
                % (fh.name, region_filter_p, region_filter_n, N),
                file=sys.stderr)

    try:
        from cpv import manhattan
        regions = manhattan.read_regions(fh.name)

        manhattan.manhattan(prefix + ".slk.bed.gz", 3, prefix.rstrip(".") + ".manhattan.png",
                         False, ['#959899', '#484B4C'], "", False, None,
                         regions=regions, bonferonni=False)
    except ImportError:
        pass # they dont have matplotlib


    if db is not None:
        from cruzdb import Genome
        g = Genome(db)
        lastf = fh.name
        with open(prefix + ".anno.%s.bed" % db, "w") as fh:
            fh.write('#')
            g.annotate(lastf, (table, "cpgIslandExt"), out=fh,
                    feature_strand=True, parallel=len(spvals) > 500)
        print("wrote: %s annotated with %s %s" % (fh.name, db, table), file=sys.stderr)


# stepsize.py
def stepsize(bed_files, col):

    D1 = []
    for bed_file in bed_files:
        for _, chromlist in groupby(bediter(bed_file, col), itemgetter('chrom')):
            L = list(chromlist)
            if len(L) < 2: continue

            last_start = 0
            for i, ibed in enumerate(L):
                assert ibed['start'] >= last_start
                # look around ibed. nearest could be up or down-stream
                if i + 2 == len(L): break
                D1.append(L[i + 1]['start'] - ibed['start'])
        # round up to the nearest 10
    return int(round(np.median(D1) + 5, -1))


#acf.py
def acf(fnames, lags, col_num0, partial=True, simple=False, mlog=True):
    """
    calculate the correlation of the numbers in `col_num0` from the bed files
    in `fnames` at various lags. The lags are specified by distance. Partial
    autocorrelation may be calculated as well.
    Since the bed files may be very large, this attempts to be as memory
    efficient as possible while still being very fast for a pure python
    implementation.
    """
    # reversing allows optimization below.
    imap = get_map()

    arg_list = [] # chaining
    for fname in fnames:
        # groupby chromosome.
        arg_list = chain(arg_list, ((list(chromlist), lags) for chrom, \
                    chromlist in \
                    groupby(bediter(fname, col_num0), lambda a: a["chrom"])))
        print('DEBUG arg_list', arg_list)

    unmerged_acfs = [] # separated by chrom. need to merge later.
    for chrom_acf in imap(_acf_by_chrom, arg_list):
        unmerged_acfs.append(chrom_acf)

    acfs = merge_acfs(unmerged_acfs)
    acf_res = {}
    xs = np.array([], dtype='f')
    ys = np.array([], dtype='f')
    # iterate over it backwards and remove to reduce memory.
    while len(acfs):
        lmin, lmax, xys = acfs.pop()
        if partial:
            xs, ys = np.array(xys["x"]), np.array(xys["y"])
        else:
            # add the inner layers as we move out.
            xs = np.hstack((xs, xys["x"]))
            ys = np.hstack((ys, xys["y"]))
        if len(xs) == 0:
            print("no values found at lag: %i-%i. skipping" \
                    % (lmin, lmax), file=sys.stderr)
            continue
        if mlog:
            xs[xs == 0] = 1e-12
            ys[ys == 0] = 1e-12
            xs, ys = -np.log10(xs), -np.log10(ys)
        #slope, intercept, corr, p_val, stderr = ss.linregress(xs, ys)
        # NOTE: using pearson correlation, which assumes normality.
        # could switch to spearman as below.
        corr, p_val = ss.spearmanr(xs, ys)
        if simple:
            acf_res[(lmin, lmax)] = corr
        else:
            acf_res[(lmin, lmax)] = (corr, len(xs), p_val)
    return sorted(acf_res.items())

def merge_acfs(unmerged):
    """
    utitlity function to merge the chromosomes after
    they've been calculated, and before the correlation
    is calculated.
    """
    merged = unmerged.pop()
    for um in unmerged:
        # have to merge at each lag.
        for (glag_min, glag_max, gxys), (ulag_min, ulag_max, uxys) in \
                                                izip(merged, um):
            assert glag_min == ulag_min and glag_max == ulag_max
            gxys["x"].extend(uxys["x"])
            gxys["y"].extend(uxys["y"])
            # reduce copies in memory.
            uxys = {}
    return merged


def create_acf_list(lags):
    acfs = []
    if len(lags) == 1:
        lags.append(lags[0])
    for lag_min, lag_max in pairwise(lags):
        acfs.append((lag_min, lag_max,
            # array uses less memory than list.
            {"x": array("f"), "y": array("f")}))
    acfs.reverse()
    return acfs

def _acf_by_chrom(args):
    """
    calculate the ACF for a single chromosome
    chromlist is the data for a single chromsome
    """
    chromlist, lags = args
    acfs = create_acf_list(lags)
    if not isinstance(chromlist, list):
        chromlist = list(chromlist)
    max_lag = max(a[1] for a in acfs)
    for ix, xbed in enumerate(chromlist):
        # find all lines within lag of xbed.
        for iy in xrange(ix + 1, len(chromlist)):
            ybed = chromlist[iy]
            # y is always > x so dist calc is simplified.
            dist = ybed['start'] - xbed['end']
            if dist > max_lag: break

            for lag_min, lag_max, xys in acfs:
                # can break out of loop because we reverse-sorted acfs
                # above. this is partial, but we merge below if needed.
                if lag_min <= dist < lag_max:
                    xys["x"].append(xbed['p'])
                    xys["y"].append(ybed['p'])
                elif dist > lag_max:
                    break
    return acfs


def merge_acfs(unmerged):
    """
    utitlity function to merge the chromosomes after
    they've been calculated, and before the correlation
    is calculated.
    """
    merged = unmerged.pop()
    for um in unmerged:
        # have to merge at each lag.
        for (glag_min, glag_max, gxys), (ulag_min, ulag_max, uxys) in \
                                                izip(merged, um):
            assert glag_min == ulag_min and glag_max == ulag_max
            gxys["x"].extend(uxys["x"])
            gxys["y"].extend(uxys["y"])
            # reduce copies in memory.
            uxys = {}
    return merged

#fdr.py
def obs_fdr(fbed_file, col_num, col_null=None):
    ps = [b['p'] for b in bediter(fbed_file, col_num)]
    if col_null is None:
        # Benjamini-Hochberg.
        nulls = np.arange(1, len(ps) + 1, dtype=np.float64) / float(len(ps))
    else:
        nulls = [b['p'] for b in bediter(fbed_file, col_null)]
    fh = ts.nopen(fbed_file)
    drop_header(fh)
    for qval, l in izip(relative_fdr(ps, nulls), fh):
        yield qval, l

def relative_fdr(observed, null):
    observed = np.asarray(observed)

    null = np.asarray(null)
    null.sort()

    obs_sort_ind = np.argsort(observed)
    observed = observed[obs_sort_ind]
    obs_unsort_ind = obs_sort_ind.argsort()

    corrected = observed / null
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected[corrected > 1] = 1
    return corrected[obs_unsort_ind]

# peaks.py
def peaks(fbedfile, col_num, threshold, seed, dist, fout, scmp):
    chromiter = peaks_bediter(fbedfile, col_num)
    for _ in walk_peaks(chromiter, threshold, seed, dist, fout, scmp):
        yield _

def peaks_bediter(fname, col_num):
    for i, l in enumerate(reader(fname, header=False)):
        if l[0][0] == "#": continue
        try:
            yield  {"chrom": l[0], "start": int(l[1]), "end": int(l[2]),
                "p": float(l[col_num])} # "stuff": l[3:][:]}
        except:
            print(l, file=sys.stderr)
            if i != 0:
                raise

# TODO use class to keep track of written peaks.
def write_peaks(peaks, seed, out, scmp):
    # could have a list with only those passing the threshold.
    if len(peaks) == 0: return None
    # peak_count unused...
    peak_start = peaks[0]["start"]
    # dont konw the length of the regions and they are only sorted
    # by start.
    peak_end = max(p["end"] for p in peaks)
    peak_count = len(peaks)
    # TODO: something better than keep best p-value ? truncated product?
    pbest = peaks[0]["p"]
    for p in peaks:
        if scmp(p["p"], pbest): pbest = p["p"]
    out.write("%s\t%i\t%i\t%.4g\t%i\n" % (
        peaks[0]["chrom"], peak_start, peak_end, pbest, peak_count))

def trim_peaks(peaks, seed, thresh, scmp):
    """
    if thresh was greater than seed, we trim the region
    so the ends are < seed, but middle values can be seed < p < thresh
    """
    if seed == thresh: return peaks
    try:
        i_start = next(i for i, p in enumerate(peaks) if scmp(p['p'], seed))
    except StopIteration:
        return []
    i_end = len(peaks) - next(i for i, p in enumerate(reversed(peaks)) if scmp(p['p'], seed))
    return peaks[i_start:i_end]

def walk_peaks(chromiter, thresh, seed, dist, out=None, scmp=operator.le):
    assert(scmp(seed, thresh))
    for key, bedlist in groupby(chromiter, lambda c: c["chrom"]):
        last_start = -1
        # have to track max end because intervals are sorted only by start.
        max_end, peaks = 0, []
        for b in bedlist:
            assert last_start <= b["start"], ("enforce sorted", last_start, b)
            last_start = b["start"]
            # this comparison gets both thresh and seed.
            if scmp(b["p"], thresh):
                # we have to add this to peaks.
                # first check distance.
                # if distance is too great, we create a new peak
                if peaks != [] and b["start"] - max_end > dist:

                    peaks = trim_peaks(peaks, seed, thresh, scmp)
                    if out is None:
                        for p in peaks: yield p
                    else:
                        write_peaks(peaks, seed, out, scmp)
                    peaks = []
                    max_end = 0

                #add new peak regardless
                peaks.append(b)
                max_end = max(b['end'], max_end)

        if out is None:
           if any(scmp(p['p'], seed) for p in peaks):
               for p in peaks: yield p
        else:
            write_peaks(peaks, seed, out, scmp)


# region-q.py
def gen_correlated(sigma, n, observed=None):
    """
    generate autocorrelated data according to the matrix
    sigma. if X is None, then data will be sampled from
    the uniform distibution. Otherwise, it will be sampled
    from X. Where X is then *all* observed
    p-values.
    """
    C = np.matrix(chol(sigma))
    if observed is None:
        X = np.random.uniform(0, 1, size=(n, sigma.shape[0]))
    else:
        assert n * sigma.shape[0] < observed.shape[0]
        idxs = np.random.random_integers(0, len(observed) - 1,
                                         size=sigma.shape[0] * n)
        X = observed[idxs].reshape((n, sigma.shape[0]))

    Q = np.matrix(qnorm(X))
    for row in  np.array(1 - norm.sf((Q * C).T)).T:
        yield row

def sl_sim(sigma, ps, nsims, sample_distribution=None):
    N = 0
    print("nsims:", nsims, file=sys.stderr)
    w0 = stouffer_liptak(ps, sigma)["p"]
    # TODO parallelize here.
    for i in range(10):
        for prow in gen_correlated(sigma, nsims/10, sample_distribution):
            s = stouffer_liptak(prow, sigma)
            if not s["OK"]: 1/0
            if s["p"] <= w0: N += 1

    return N / float(nsims)

def _gen_acf(region_info, fpvals, col_num, step):
    # calculate the ACF as far out as needed...
    # keys of region_info are (chrom, start, end)
    max_len = max(int(r[2]) - int(r[1]) for r in region_info)
    print("# calculating ACF out to: %i" % max_len, file=sys.stderr)

    lags = list(range(1, max_len, step))
    if len(lags) == 0:
        lags.append(max_len)

    if lags[-1] < max_len: lags.append(lags[-1] + step)
    if len(lags) > 20:
        repr_lags = "[" + ", ".join(map(str, lags[1:4])) + \
                    " ... " + \
                    ", ".join(map(str, lags[-5:])) + "]"
    else:
        repr_lags = str(lags)
    print("#           with %-2i lags: %s" \
            % (len(lags), repr_lags), file=sys.stderr)

    if len(lags) > 100:
        print("# !! this could take a looong time", file=sys.stderr)
        print("# !!!! consider using a larger step size (-s)", file=sys.stderr)
    acfs = acf(fpvals, lags, col_num, simple=True)
    print("# Done with one-time ACF calculation", file=sys.stderr)
    return acfs

def get_total_coverage(fpvals, col_num, step, out_val):
    """
    Calculate total bases of coverage in `fpvals`.
    Used for the sidak correction
    """
    total_coverage = 0
    for key, chrom_iter in groupby(bediter(fpvals, col_num),
            itemgetter('chrom')):
        bases = set([])
        for feat in chrom_iter:
            s, e = feat['start'], feat['end']
            if s == e: e += 1
            #e = max(e, s + step)
            bases.update(range(s, e))
        total_coverage += len(bases)
    out_val.value = total_coverage

def _get_total_coverage(fpvals, col_num, step):
    from multiprocessing import Process, Value
    val = Value('f')
    p = Process(target=get_total_coverage, args=(fpvals, col_num, step, val))
    p.start()
    return p, val

def sidak(p, region_length, total_coverage, message=[False]):
    """
    see: https://github.com/brentp/combined-pvalues/issues/2
    """
    if region_length == 0:
        region_length = 1
        if not message[0]:
            message[0] = True
            sys.stderr.write(""""warning: 0-length region found.
does input have 0-length intervals? using length of 1 and not reporting
further 0-length intervals""")
    # use 1.1 as heuristic to account for limit in available regions
    # of a given size as the region_length increases
    # TODO: base that on the actual number of regiosn of this length
    # that could be seen based on the distance constraint.
    k = total_coverage / (np.float64(region_length)**1.0)
    if k < 1: k = total_coverage
    p_sidak = 1 - (1 - p)**k
    if p_sidak == 0:
        assert p < 1e-16, (p, k, total_coverage, region_length)
        p_sidak = (1 - (1 - 1e-16)**k) / (p / 1e-16)
        p_sidak = min(p_sidak, p * k)

    # print "bonferroni:", min(p * k, 1)
    return min(p_sidak, 1)

def _get_ps_in_regions(tree, fpvals, col_num):
    """
    find the pvalues associated with each region
    """
    region_info = defaultdict(list)
    for row in bediter(fpvals, col_num):
        for region in tree[row['chrom']].find((row['start'], row['end'])):
            region_len = max(1, region[1] - region[0])
            region_tup = tuple(region[-1])
            region_info[region_tup].append(row)
    assert sum(len(v) for v in tree.values()) >= len(region_info)
    if sum(len(v) for v in tree.values()) > len(region_info):
        sys.stderr.write("# note: not all regions contained measurements\n")
    return region_info

def read_regions(fregions):
    tree = defaultdict(InterLap)
    for i, toks in enumerate(ts.reader(fregions, header=False)):
        if i == 0 and not (toks[1] + toks[2]).isdigit(): continue
        tree[toks[0]].add((int(toks[1]), int(toks[2]), toks))
    sys.stderr.write("# read %i regions from %s\n" \
            % (sum(len(v) for v in tree.values()), fregions))
    return tree

def region_p(fpvals, fregions, col_num, step, z=True):
    # just use 2 for col_num, but dont need the p from regions.

    tree = read_regions(fregions)
    process, total_coverage_sync = _get_total_coverage(fpvals, col_num, step)

    region_info = _get_ps_in_regions(tree, fpvals, col_num)

    acfs = _gen_acf(region_info, (fpvals,), col_num, step)
    process.join()
    total_coverage = total_coverage_sync.value

    # regions first and then create ACF for the longest one.
    print("%i bases used as coverage for sidak correction" % \
                                (total_coverage), file=sys.stderr)
    sample_distribution = np.array([b["p"] for b in bediter(fpvals,
                                                                col_num)])

    combine = z_score_combine if z else stouffer_liptak
    for region, prows in region_info.items():
        # gen_sigma expects a list of bed dicts.
        sigma = gen_sigma_matrix(prows, acfs)
        ps = np.array([prow["p"] for prow in prows])
        if ps.shape[0] == 0:
            print("bad region", region, file=sys.stderr)
            continue

        # calculate the SLK for the region.
        region_slk = combine(ps, sigma)
        if not region_slk["OK"]:
            print("problem with:", region_slk, ps, file=sys.stderr)

        slk_p = region_slk["p"]

        sidak_slk_p = sidak(slk_p, int(region[2]) - int(region[1]), total_coverage)

        result = ["\t".join(region), slk_p, sidak_slk_p, "NA"]
        yield result


# filter.py

def ilogit(v):
    return 1 / (1 + exp(-v))

def fix_bed(fname):
    """
    a lot of bed files will have no header or have e.g.
    8e6 instead of 8000000 for start/end. this just fixes that
    so we can send to bedtools
    """
    r = ts.reader(fname, header=False)
    h = next(r)
    assert not (h[1] + h[2]).isdigit(), "must have header for filtering"
    tname = mktemp()
    fh = ts.nopen(tname, "w")
    fh.write("#" + "\t".join(h) + "\n")
    for toks in r:
        toks[1:3] = map(str, (int(float(t)) for t in toks[1:3]))
        fh.write("%s\n" % "\t".join(toks))
    fh.close()
    atexit.register(os.unlink, tname)
    return tname

def bed_filter(p_bed, region_bed, max_p=None, region_p=None, p_col_name="P.Value",
                    coef_col_name="logFC"):

    ph = ts.header(p_bed)
    if (ph[1] + ph[2]).isdigit():
        raise Exception('need header in p-value file to run filter')
    assert ph[1] == 'start' and ph[2] == 'end' and ph[0] == 'chrom', \
            ('must have chrom, start, end header for', p_bed)
    ph = ['p' + h for h in ph]

    rh = ts.header(region_bed)
    header = not (rh[1] + rh[2]).isdigit()

    if isinstance(p_col_name, str) and p_col_name.isdigit():
        p_col_name = int(p_col_name) - 1

    if isinstance(p_col_name, (int, long)):
        p_col_name = ph[p_col_name][1:]

    a = dict(p_bed=p_bed, region_bed=region_bed)
    a['p_bed'] = fix_bed(a['p_bed'])
    a['header'] = ""

    j = 0
    for group, plist in groupby(
            ts.reader('|bedtools intersect -b %(p_bed)s \
                         -a %(region_bed)s -wo %(header)s' % a,
            header=rh + ph), itemgetter('chrom','start','end')):
        plist = list(plist)

        if region_p:
            r = plist[0] # first cols are all the same
            region_p_key = 'slk_sidak_p' if 'slk_sidak_p' in r \
                                         else 'z_sidak_p' if 'z_sidak_p' in r \
                                         else None
            if region_p_key is None: raise Exception
            if float(r[region_p_key]) > region_p:
                continue

        try:
            plist = [x for x in plist if (int(x['start']) <= int(x['pstart']) <= int(x['pend'])) and ((int(x['start']) <= int(x['pend']) <= int(x['end'])))]
        except:
            print(plist)
            raise
        tscores = [float(row['pt']) for row in plist if 'pt' in row]

        if max_p:
            if any(float(row['p' + p_col_name]) > max_p for row in plist):
                continue

        ngt05  = sum(1 for row in plist if float(row['p' + p_col_name]) > 0.05)

        # logic to try to find t and coef headers and skip if not found
        extra_header = []
        extra = []
        if tscores:
            tpos = sum(1 for ts in tscores if ts > 0)
            tneg = sum(1 for ts in tscores if ts < 0)
            tpn = "%i/%i" % (tpos, tneg)

            tsum = str(sum(ts for ts in tscores))
            extra_header += ["t.pos/t.neg", "t.sum"]
            extra += [tpn, tsum]
        else:
            tsum = tpn = "NA"
        if 'p' + coef_col_name not in plist[0] and 'pcoefficient' in plist[0]:
            coef_col_name = 'coefficient'
        if 'p' + coef_col_name in plist[0]:
            coef = (sum(float(row['p' + coef_col_name]) for row in plist) /
                                    len(plist))

            # since we probably had the data logit transformed, here we
            # do the inverse and subtract 0.5 since ilogit(0) == 0.5
            icoef = (sum(ilogit(float(row['p' + coef_col_name])) for row in plist) /
                                    len(plist)) - 0.5
            extra_header += ["avg.diff", "ilogit.diff"]
            extra += ["%.3f" % coef, "%.3f" % icoef]
        else:
            coef = icoef = float('nan')

        frow = [plist[0][h] for h in rh] + extra
        if j == 0:
            yield rh + extra_header
            j = 1
        yield frow


def genomic_control(pvals):
    """
    calculate genomic control factor, lambda
    >>> genomic_control([0.25, 0.5, 0.75])
    1.0000800684096998
    >>> genomic_control([0.025, 0.005, 0.0075])
    15.715846578113579
    """
    from scipy import stats
    import numpy as np
    pvals = np.asarray(pvals)
    return np.median(stats.chi2.ppf(1 - pvals, 1)) / 0.4549

def genome_control_adjust(pvals):
    """
    adjust p-values by the genomic control factor, lambda
    >>> genome_control_adjust([0.4, 0.01, 0.02])
    array([ 0.8072264 ,  0.45518836,  0.50001716])
    """
    import numpy as np
    from scipy import stats
    pvals = np.asarray(pvals)
    qchi = stats.chi2.ppf(1 - pvals, 1)
    gc = np.median(qchi) / 0.4549
    return 1 - stats.chi2.cdf(qchi / gc, 1)


def get_col_num(c, bed_file=None):
    """
    adjust the col number so it does intutive stuff
    for command-line interface
    >>> get_col_num(4)
    3
    >>> get_col_num(-1)
    -1
    """
    if isinstance(c, basestring) and c.isdigit():
        c = int(c)
    if isinstance(c, (int, long)):
        return c if c < 0 else (c - 1)
    header = ts.header(bed_file)
    assert c in header
    return header.index(c)


def bediter(fnames, col_num, delta=None):
    """
    iterate over a bed file. turn col_num into a float
    and the start, stop column into an int and yield a dict
    for each row.
    """
    last_chrom = chr(0)
    last_start = -1
    if isinstance(fnames, basestring):
        fnames = [fnames]
    for fname in fnames:
        for i, l in enumerate(ts.reader(fname, header=False)):
            if l[0][0] == "#": continue
            if i == 0: # allow skipping header
                try:
                    float(l[col_num])
                except ValueError:
                    continue
            chrom = l[0]
            start = int(float(l[1]))
            if chrom == last_chrom:
                assert start >= last_start, ("error at line: %i, %s"
                        % (i, "\t".join(l)), "file is not sorted")
            else:
                assert last_chrom < chrom, ("error at line: %i, %s "
                        " with file: %s" % (i, "\t".join(l), fname),
                        "chromosomes must be sorted as characters",
                        last_chrom, "is not < ", chrom)
                last_chrom = chrom

            last_start = start

            p = float(l[col_num])
            if not delta is None:
                if p > 1 - delta: p-= delta # the stouffer correction doesnt like values == 1
                if p < delta: p = delta # the stouffer correction doesnt like values == 0

            v = {"chrom": l[0], "start": start, "end": int(float(l[2])),
                 "p": p} # "stuff": l[3:][:]}
            if v['end'] - v['start'] > 100000:
                print("warning! large interval at %s will increase memory use." % v)
            yield v


# slk.py

def get_corr(dist, acfs):
    """
    extract the correlation from the acf sigma matrix
    given a distance.
    """
    # it's very close. just give it the next up.
    # TODO: should probably not do this. force them to start at 0.
    # acfs[0] is like (lag_min, lag_max), corr
    # so this is checking if it's < smallest lag...
    if dist < acfs[0][0][0]:
        return acfs[0][1]
    for (lag_min, lag_max), corr in acfs:
        if lag_min <= dist <= lag_max:
            return corr
    return 0

def walk_slk(chromlist, lag_max):
    """
    for each item in chromlist, yield the item and its neighborhood
    within lag-max. These yielded values are then used to generate
    the sigma autocorrelation matrix.
    """
    L = list(chromlist) if not isinstance(chromlist, list) else chromlist

    N = len(L)
    imin = imax = 0
    for ithis, xbed in enumerate(L):
        # move up the bottom of the interval
        while xbed["start"] - L[imin]["end"] > lag_max:
            imin += 1
        if imax == N: imax -= 1
        while L[imax]["start"] - xbed["end"] < lag_max:
            imax += 1
            if imax == N: break
        assert imin <= ithis <= imax
        # dont need to add 1 to imax because we got outside of the range above.
        yield xbed, L[imin: imax]

def gen_sigma_matrix(group, acfs, cached={}):
    a = np.eye(len(group), dtype=np.float64)
    group = enumerate(group)
    for (i, ibed), (j, jbed) in combinations(group, 2):
        # j is always right of i. but could overlap
        dist = jbed["start"] - ibed["end"]
        if dist < 0: dist = 0
        # symmetric.
        # cached speeds things up a bit...
        if not dist in cached:
            cached[dist] = get_corr(dist, acfs)
        a[j, i] = a[i, j] = cached[dist]

    return a

def slk_chrom(chromlist, lag_max, acfs, z=True):
    """
    calculate the slk for a given chromosome
    """
    arr = np.empty((len(chromlist),),  dtype=np.dtype([
        ('start', np.uint32),
        ('end', np.uint32),
        ('p', np.float32),
        ('slk_p', np.float32)]))

    for i, (xbed, xneighbors) in enumerate(walk_slk(chromlist, lag_max)):

        sigma = gen_sigma_matrix(xneighbors, acfs)
        pvals = np.array([g['p'] for g in xneighbors])
        r = z_score_combine(pvals, sigma)
        arr[i] = (xbed["start"], xbed["end"], xbed["p"], r["p"])
    return xbed['chrom'], arr

def _slk_chrom(args):
    return slk_chrom(*args)

def adjust_pvals(fnames, col_num0, acfs, z=True):
    lag_max = acfs[-1][0][1]

    # parallelize if multiprocesing is installed.
    imap = get_map()
    arg_iter = []
    for fname in fnames:
        # 9e-17 seems to be limit of precision for cholesky.
        arg_iter = chain(arg_iter, ((list(chromlist), lag_max, acfs,
            z) \
                    for key, chromlist in groupby(bediter(fname, col_num0, 9e-117),
                            itemgetter("chrom"))))

    for chrom, results in imap(_slk_chrom, arg_iter):
        yield chrom, results


def stouffer_liptak(pvals, sigma=None):
    """
    The stouffer_liptak correction.
    >>> stouffer_liptak([0.1, 0.2, 0.8, 0.12, 0.011])
    {'p': 0.0168..., 'C': 2.1228..., 'OK': True}
    >>> stouffer_liptak([0.5, 0.5, 0.5, 0.5, 0.5])
    {'p': 0.5, 'C': 0.0, 'OK': True}
    >>> stouffer_liptak([0.5, 0.1, 0.5, 0.5, 0.5])
    {'p': 0.28..., 'C': 0.57..., 'OK': True}
    >>> stouffer_liptak([0.5, 0.1, 0.1, 0.1, 0.5])
    {'p': 0.042..., 'C': 1.719..., 'OK': True}
    >>> stouffer_liptak([0.5], np.matrix([[1]]))
    {'p': 0.5...}
    """
    L = len(pvals)
    pvals = np.array(pvals, dtype=np.float64)
    pvals[pvals == 1] = 1.0 - 9e-16
    qvals = norm.isf(pvals, loc=0, scale=1).reshape(L, 1)
    if any(np.isinf(qvals)):
        raise Exception("bad values: %s" % pvals[list(np.isinf(qvals))])

    # dont do the correction unless sigma is specified.
    result = {"OK": True}
    if not sigma is None:
        try:
            C = chol(sigma)
            Cm1 = np.asmatrix(C).I # C^-1
            # qstar
            qvals = Cm1 * qvals
        except LinAlgError as e:
            result["OK"] = False
            result = z_score_combine(pvals, sigma)
            return result

    Cp = qvals.sum() / np.sqrt(len(qvals))
    # get the right tail.
    pstar = norm.sf(Cp)
    if np.isnan(pstar):
        print("BAD:", pvals, sigma, file=sys.stderr)
        pstar = np.median(pvals)
        result["OK"] = True
    result.update({"C": Cp, "p": pstar})
    return result

def z_score_combine(pvals, sigma):
    L = len(pvals)
    pvals = np.array(pvals, dtype=np.float64)
    pvals[pvals == 1] = 1.0 - 9e-16
    z = np.mean(norm.isf(pvals, loc=0, scale=1))
    sz = 1.0 /L * np.sqrt(L + 2 * np.tril(sigma, k=-1).sum())
    res = {'p': norm.sf(z/sz), 'OK': True}
    return res

def fisherp(pvals):
    """ combined fisher probability without correction """
    s = -2 * np.sum(np.log(pvals))
    return chisqprob(s, 2 * len(pvals))

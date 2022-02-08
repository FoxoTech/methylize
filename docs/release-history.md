# Release History

## v1.0.1
- Differentially methylated regions (DMR) will no annotate via cached UCSC database (via fetch_genes) without using the internet, if user wants. (previously, it would still contact the internet database even if user told it not to)
- Added testing via github actions, and increased speed
- updated documentation

## v1.0.0
  - fixed bug in fetch_genes() from UCSC browser; function will now accept either the filepath or the DMR dataframe output.

## v0.9.9
- Added a differentially methylated regions (DMR) functions that takes the output of the `diff_meth_pos` (DMP) function.
  - DMP maps differences to chromosomes; DMR maps differences to specific genomic locii, and requires more processing.
  - upgraded methylprep manifests to support both `old` and `new` genomic build mappings for all array types.
    In general, you can supply a keyword argument (`genome_build='OLD'`) to change from the new build back to the old one.
  - Illumina 27k arrays are still not supported, but mouse, epic, epic+, and 450k ARE supported.
    (Genome annotation won't work with `mouse` array, only human builds.)
  - DMP integrates the `combined-pvalues` package (https://pubmed.ncbi.nlm.nih.gov/22954632/)
  - DMP integrates with UCSC Genome (refGene) and annotates the genes near CpG regions.
  - Annotation includes column(s) showing the   tissue specific expression levels of relevant genes (e.g. `filter=blood`)
  this function is also available with extended options as `methylize.filter_genes()`
  - provides output BED and CSV files for each export into other genomic analysis tools
- `methylize.to_BED` will convert the diff_meth_pos() stats output into a standard BED file
  (a tab separated CSV format with standardized, ordered column names)

## v0.9.8
- fixed methylize diff_meth_pos linear regression. upgraded features too
  - Fixed bug in diff_meth_pos using linear regression - was not calculating p-values correctly.
    Switched from statsmodels OLS to scipy linregress to fix, but you can use either one with kwargs.
    They appear to give exactly the same results now after testing.
  - The "CHR-" prefix is omitted from manhattan plots by default now
  - dotted manhattan sig line is Bonferoni corrected (pass in post_test=None to leave uncorrected)
  - added a probe_corr_plot() undocumented function, a scatterplot of probe confidence intervals vs pvalue
  - sorts probes by MAPINFO (chromosome location) instead of FDR_QValue on manhattan plots now
- Support for including/excluding sex chromosomes from DMP (probe2chr map)

## v0.9.5
- Added imputation to diff_meth_pos() function, because methylprep output contains missing values
by default and cannot be used in this function.
  - This can be disabled, and it will throw a warning if NaNs present.
  - Default is to delete probes that have any missing values before running analysis.
  - if 'auto': If there are less than 30 samples, it will delete the missing rows.
  - if 'auto': If there are >= 30 samples in the batch analyzed, it will replace NaNs with the
  average value for that probe across all samples.
  - User may override the default using: True ('auto'), 'delete', 'average', and False (disable)
  - diff_meth_pos() now support mouse array, with multiple copies of the same probe names.

## v0.9.4
- Fixed bug where methylize could not find a data file in path, causing ImportError
- Improved diff_meth_pos() function and added support for all array types. Now user must
specify the array_type when calling the function, as the input data are stats, not probe betas,
so it cannot infer the array type from this information.

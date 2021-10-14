# Release History

## v0.9.9
- Added differentially methylated regions (DMR) functions, alongside diff-meth-pos(ition) DMP functions.
  - DMP maps differences to chromosomes; DMR maps differences to specific genomic locii, and requires more processing.
  - upgraded methylprep manifests to support both old and new genomic build mappings for all array types.
  - Illumina 27k arrays are still not supported, but mouse, epic, epic+, and 450k ARE supported.
- diff_meth_position() function
  - integrates the combined-pvalues package (https://pubmed.ncbi.nlm.nih.gov/22954632/)
  - integrates with UCSC Genome (refGene) and annotates the genes near CpG regions.
  - includes columns showing the tissue specific expression levels of relevant genes (filter=blood)
  this function is also available with extended options as methylize.filter_genes()
  - provides output BED and CSV files for each export into other genomic analysis tools
- methylize.to_BED will convert the diff_meth_pos() stats output into a standard BED file
  (tab separated CSV format with standardized, ordered column names)

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

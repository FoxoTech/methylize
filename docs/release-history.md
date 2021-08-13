# Release History

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

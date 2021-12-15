`methylize` is a python package for analyzing output from Illumina methylation arrays. It complements `methylprep` and `methylcheck` and provides methods for computing differentially methylated probes and regions, and annotating these regions with the UCSC Genome Browser.  View on [ReadTheDocs.](https://life-epigenetics-methylize.readthedocs-hosted.com/en/latest/)

[![Readthedocs](https://readthedocs.com/projects/life-epigenetics-methylize/badge/?version=latest)](https://life-epigenetics-methylize.readthedocs-hosted.com/en/latest/) [![image](https://img.shields.io/pypi/l/pipenv.svg)](https://python.org/pypi/pipenv) [![CircleCI](https://circleci.com/gh/FoxoTech/methylize/tree/master.svg?style=shield)](https://circleci.com/gh/FoxoTech/methylize/tree/master) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/099d26465bd64c2387afa063810a13e6)](https://www.codacy.com/gh/FoxoTech/methylize/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=FOXOBioScience/methylize&amp;utm_campaign=Badge_Grade) [![Coverage Status](https://coveralls.io/repos/github/FoxoTech/methylize/badge.svg?branch=master)](https://coveralls.io/github/FoxoTech/methylize?branch=master) ![PYPI-Downloads](https://img.shields.io/pypi/dm/methylize.svg?label=pypi%20downloads&logo=PyPI&logoColor=white)

- [Differentially methylated position (DMP) regression, detection and visualation](docs/demo_diff_meth_pos.ipynb)
  - [Logistic Regression](docs/methylize_tutorial.html#Differentially-methylated-Regions-Analysis-with-Binary-Phenotypes)
  - [Linear Regression](docs/methylize_tutorial.html#Differentially-methylated-Regions-Analysis-with-Continuous-Numeric-Phenotypes)
  - [Manhattan Plot](docs/methylize_tutorial.html#Manhattan-Plots)
  - [Volcano plot](docs/methylize_tutorial.html#Volcano-Plot)
- [Differentially methylated regions](docs/diff_meth_regions.md)
  - [Gene annotation with the UCSC Human Genome Browser](docs/diff_meth_regions.html#gene-annotation-with-ucsc-genome-browser)

##Installation

```python
pip3 install methylize
```

Installation will also install the other parts of the `methylsuite` (methylprep and methylcheck) if they are not already installed.

If progress bar is missing:
    If you don't see a progress bar in your jupyterlab notebook, try this:

    - conda install -c conda-forge nodejs
    - jupyter labextension install @jupyter-widgets/jupyterlab-manager

##Methylize Package

The `methylize` package contains both high-level APIs for processing data from local files and low-level functionality allowing you to analyze your data AFTER running `methylprep` and `methylcheck`. For greatest usability, import `methylize` into a Jupyer Notebook along with your processed sample data (a DataFrame of beta values or m-values and a separate DataFrame containing meta data about the samples).

`Methylize` allows you to run linear or logistic regression on all probes and identify points of interest in the methylome where DNA is differentially modified. Then you can use these regression results to create *volcano plots* and *manhattan plots*.

###Sample Manhattan Plot
![Manhattan Plot](https://github.com/FoxoTech/methylize/blob/master/docs/manhattan_example.png?raw=true)

![Manhattan Plot (alternate coloring)](https://github.com/FoxoTech/methylize/blob/master/docs/manhattan_example2.png?raw=true)

###Sample Volcano Plot
![Volcano Plot](https://github.com/FoxoTech/methylize/blob/master/docs/volcano_example.png?raw=true)

Customizable: Plot size, color palette, and cutoff p-value lines can be set by the user.
Exporting: You can export all probe statistics, or just the significant probes as CSV or python pickled DataFrame.

##Differentially methylated position/probe (DMP) detection

The `diff_meth_pos(meth_data, phenotypes)` function searches for individual differentially methylated positions/probes
(DMPs) by regressing methylation `beta values` or `M-values` for each sample at a given
genomic location against the phenotype data for those samples.

###Phenotypes

Can be provided as

    - a list of strings,
    - integer binary data,
    - numeric continuous data
    - pandas series or numpy array

Only 2 phenotypes are allowed with logistic regression. Linear regression can take more than two phenotypes.
The function will coerge string labels for phenotype into 0s and 1s when running logistic regression. To use the meta data associated with a dataset, you would need to pass in the column of your meta dataframe as a list or series. The order of the items in the phenotype should match the order of samples in the beta values or M-values.

For details on all of the other adjustable input parameters, refer to the API for [diff_meth_pos()](docs/source/modules.html#module-methylize.diff_meth_pos)

###Returns
A pandas dataframe of regression `statistics` with one row for each probe
and these columns:

    - regression coefficient
    - lower limit of the coefficient's 95% confidence interval
    - upper limit of the coefficient's 95% confidence interval
    - standard error
    - p-value (phenotype group A vs B - likelihood that the difference is significant for this probe/location)
    - q-value (p-values corrected for multiple testing using the Benjamini-Hochberg FDR method)
    - FDR_QValue: p value, adjusted for multiple comparisons

    The rows are sorted by q-value in ascending order to list the most significant
    probes first. If q_cutoff is specified, only probes with significant q-values
    less than the cutoff will be returned in the dataframe.

##Differentially methylated regions
Pass in your DMP stats dataframe, and it calculates and annotates differentially methylated regions (DMR) using the `combined-pvalues` pipeline and returns list of output files.

    - calculates auto-correlation
    - combines adjacent P-values
    - performs false discovery adjustment
    - finds regions of enrichment (i.e. series of adjacent low P-values)
    - assigns significance to those regions
    - annotates significant regions with possibly relevant nearby Genes, using the UCSC Genome Browser Database
    - annotates candidate genes with expression levels for the sample tissue type, if user specifies the
    sample's tissue type.
    - returns everything in a CSV that can be imported into other Genomic analysis packages.

For more details on customizing the inputs and outputs, see API for the [diff_meth_regions(stats, array_type)](docs/source/modules.html#module-methylize.diff_meth_regions) function.

##Loading processed data

Assuming you previously used `methylprep` to process a data set like this:

```python
python -m methylprep -v process -d GSE130030 --betas
```

This creates two files, `beta_values.pkl` and `sample_sheet_meta_data.pkl`. You can load both in `methylize` like this:

Navigate to the folder where `methylrep` saved its processed files, and start a python interpreter:
```python
>>>import methylcheck
>>>data, meta = methylcheck.load_both()
INFO:methylize.helpers:loaded data (485512, 14) from 1 pickled files (0.159s)
INFO:methylize.helpers:meta.Sample_IDs match data.index (OK)
```

Or if you are running in a notebook, specify the path:
```python
import methylcheck
data, meta = methylcheck.load_both('<path_to...>/GSE105018')
```

This also validates both files, and ensures that the `Sample_ID` column in meta DataFrame aligns with the column names in the `data DataFrame`.

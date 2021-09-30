`methylize` is a python package for analyzing output from Illumina methylation arrays. It complements `methylprep` and `methylcheck`. View on [ReadTheDocs.](https://life-epigenetics-methylize.readthedocs-hosted.com/en/latest/)

[![Readthedocs](https://readthedocs.com/projects/life-epigenetics-methylize/badge/?version=latest)](https://life-epigenetics-methylize.readthedocs-hosted.com/en/latest/) [![image](https://img.shields.io/pypi/l/pipenv.svg)](https://python.org/pypi/pipenv) [![CircleCI](https://circleci.com/gh/FoxoTech/methylize/tree/master.svg?style=shield)](https://circleci.com/gh/FoxoTech/methylize/tree/master) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/099d26465bd64c2387afa063810a13e6)](https://www.codacy.com/gh/FoxoTech/methylize/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=FOXOBioScience/methylize&amp;utm_campaign=Badge_Grade) [![Coverage Status](https://coveralls.io/repos/github/FoxoTech/methylize/badge.svg?t=uf7qX4)](https://coveralls.io/github/FOXOBioScience/methylize)

1. [Overview](README.md)
2. [Demonstrating differentially methylated probe (DMP) detection (volcano plot) and mapping to chrosomes (manhattan plot)](docs/demo_diff_meth_pos.ipynb)
3. [About BumpHunter](docs/bumphunter.md)

## Methylize Package

The `methylize` package contains both high-level APIs for processing data from local files and low-level functionality allowing you to analyze your data AFTER running `methylprep` and `methylcheck`. For greatest usability, import `methylize` into a Jupyer Notebook along with your processed sample data (a DataFrame of beta values or m-values and a separate DataFrame containing meta data about the samples).

`Methylize` allows you to run linear or logistic regression on all probes and identify points of interest in the methylome where DNA is differentially modified. Then you can use these regression results to create *volcano plots* and *manhattan plots*.

### Sample Manhattan Plot
![Manhattan Plot](https://github.com/FoxoTech/methylize/blob/master/docs/manhattan_example.png?raw=true)

![Manhattan Plot (alternate coloring)](https://github.com/FoxoTech/methylize/blob/master/docs/manhattan_example2.png?raw=true)

### Sample Volcano Plot
![Volcano Plot](https://github.com/FoxoTech/methylize/blob/master/docs/volcano_example.png?raw=true)

Customizable: Plot size, color palette, and cutoff p-value lines can be set by the user.
Exporting: You can export all probe statistics, or just the significant probes as CSV or python pickled DataFrame.

## Installation

```python
pip install methylize
```

## differentially methylated position/probe (DMP) detection

The `diff_meth_pos()` function searches for individual differentially methylated positions/probes
(DMPs) by regressing the methylation M-value for each sample at a given
genomic location against the phenotype data for those samples.

Phenotypes can be provided as
  - a list of string-based,
  - integer binary data,
  - numeric continuous data
  - (TODO: use the methylprep generated meta-data dataframe as input)

The function will coerge string labels for phenotype into 0s and 1s when running logistic regression.
Only 2 phenotypes are allowed with logistic regression. Linear regression can take more than two phenotypes.

### Inputs and Parameters
-------------------------

    meth_data:
        A pandas dataframe of methylation M-values for
        where each column corresponds to a CpG site probe and each
        row corresponds to a sample.
    pheno_data:
        A list or one dimensional numpy array of phenotypes
        for each sample row in meth_data.
        - Binary phenotypes can be presented as a list/array
        of zeroes and ones or as a list/array of strings made up
        of two unique words (i.e. "control" and "cancer"). The first
        string in phenoData will be converted to zeroes, and the
        second string encountered will be convered to ones for the
        logistic regression analysis.
        - Use numbers for phenotypes if running linear regression.
    regression_method: (logistic | linear)
        - Either the string "logistic" or the string "linear"
        depending on the phenotype data available.
        - Default: "linear"
        - Phenotypes with only two options (e.g. "control" and "cancer") can be analyzed with a logistic regression
        - Continuous numeric phenotypes (e.g. age) are required to run a linear regression analysis.
    q_cutoff:
        - Select a cutoff value to return only those DMPs that meet a
        particular significance threshold. Reported q-values are
        p-values corrected according to the model's false discovery
        rate (FDR).
        - Default: 1 -- returns all DMPs regardless of significance.
    export:
        - default: False
        - if True or 'csv', saves a csv file with data
        - if 'pkl', saves a pickle file of the results as a dataframe.
        - USE q_cutoff to limit what gets saved to only significant results.
            by default, q_cutoff == 1 and this means everything is saved/reported/exported.
    filename:
        - specify a filename for the exported file.
        By default, if not specified, filename will be `DMP_<number of probes in file>_<number of samples processed>_<current_date>.<pkl|csv>`
    shrink_var:
        - If True, variance shrinkage will be employed and squeeze
        variance using Bayes posterior means. Variance shrinkage
        is recommended when analyzing small datasets (n < 10).
        (NOT IMPLEMENTED YET)

### Returns

    A pandas dataframe of regression statistics with a row for each probe analyzed
    and columns listing the individual probe's regression statistics of:
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

If Progress Bar Missing:
    if you don't see a progress bar in your jupyterlab notebook, try this:

    - conda install -c conda-forge nodejs
    - jupyter labextension install @jupyter-widgets/jupyterlab-manager


### Loading processed data

Assuming you previously used `methylprep` to process a data set like this:

```python
python -m methylprep -v process -d GSE130030 --betas
```

This creates two files, `beta_values.pkl` and `sample_sheet_meta_data.pkl`. You can load both in `methylize` like this:

Navigate to the `GSE130030` folder created by `methylrep`, and start a python interpreter:
```python
>>>import methylize
>>>data,meta = methylize.load_both()
INFO:methylize.helpers:loaded data (485512, 14) from 1 pickled files (0.159s)
INFO:methylize.helpers:meta.Sample_IDs match data.index (OK)
```
Or if you are running in a notebook, specify the full path:
```python
import methylize
data,meta = methylize.load_both('<path_to...>/GSE105018')
```

This also validates both files, and ensures that the `Sample_ID` column in meta DataFrame aligns with the column names in the `data DataFrame`.

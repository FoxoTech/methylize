## differentially methylated regions (DMR) in methylize
### adopted from `combined-pvalues` package

ref: "Comb-p: software for combining, analyzing, grouping and correcting spatially correlated P-values" [doi: 10.1093/bioinformatics/bts545](https://pubmed.ncbi.nlm.nih.gov/22954632/)

There are many similar packages

### What are Differentially methylated regions (DMRs)?
Genomic regions where DNA methylation levels differ between two groups of samples. DNA methylation is associated with cell differentiation, regulation, and proliferation, so these regions indicate that nearby genes may be involved in transcription regulation. There are several different types of DMRs. These include:

- tissue-specific DMR (tDMR),
- cancer-specific DMR (cDMR),
- development stages (dDMRs),
- reprogramming-specific DMR (rDMR),
- allele-specific DMR (AMR),
- aging-specific DMR (aDMR).

### How do run DMR

First, assuming you have processed data using `methylprep`, use `methylize` to convert a dataframe of beta or M-values into differentially-methylated-probe (DMP) statistics, using `methylize.diff_meth_pos`. You will need to provide the data along with a list of sample labels for how to separate the data into two (treatment / control) groups or more groups (levels of a phenotypic characteristic, such as age or BMI):

```
meth_data = pd.read_pickle('beta_values.pkl')
phenotypes = ["fetal","fetal","fetal","adult","adult","adult"]
test_results = methylize.diff_meth_pos(meth_data, phenotypes)
```

`phenotypes` can be a list, numpy array, or pandas Series (Anything list-like). The results will be a dataframe with p-values, a measure of the the likelihood that each probe is significantly different between the phenotypic groups. The lower the p-value, the more likely it is that groups differ:

```
IlmnID                       Coefficient  StandardError        PValue  95%CI_lower  95%CI_upper  Rsquared  FDR_QValue
cg04680738_II_R_C_rep1_EPIC     -0.03175       0.001627  1.171409e-06    -0.992267    -0.992168  0.984496    0.009706
cg18340948_II_R_C_rep1_EPIC      0.10475       0.005297  1.084911e-06     0.992256     0.992570  0.984887    0.009706
cg03681905_II_F_C_rep1_EPIC     -0.04850       0.002141  4.843320e-07    -0.994254    -0.994157  0.988444    0.009706
cg01815889_II_F_C_rep1_EPIC     -0.05075       0.002735  1.579625e-06    -0.991492    -0.991308  0.982875    0.009816
cg05995891_II_R_C_rep1_EPIC     -0.04050       0.002407  2.812417e-06    -0.989670    -0.989474  0.979254    0.013981
...                                  ...            ...           ...          ...          ...       ...         ...
cg05855048_II_R_C_rep1_EPIC      0.00025       0.016362  9.883052e-01    -0.025827     0.038289  0.000039    0.999077
cg23130711_II_R_C_rep1_EPIC      0.00025       0.016530  9.884235e-01    -0.026217     0.038553  0.000038    0.999156
cg10163088_II_F_C_rep1_EPIC     -0.00025       0.017168  9.888530e-01    -0.039573     0.027696  0.000035    0.999550
cg04079257_II_F_C_rep1_EPIC     -0.00025       0.017430  9.890214e-01    -0.039997     0.028300  0.000034    0.999679
cg24902557_II_F_C_rep1_EPIC      0.00025       0.017533  9.890857e-01    -0.028535     0.040163  0.000034    0.999704
```

The FDR_QValue is the key result. FDR_Q is the "False Discovery Rate Q-value": The adjustment corrects individual probe p-values for the number of repeated tests (once for each probe on the array).

Next, you take this whole dataframe output from `diff_meth_pos` and feed it into the DMR function, `diff_meth_regions`, along with the type of methylation array you are using:

```
manifest_or_array_type = '450k'
files_created = methylize.diff_meth_regions(stats_results, manifest_or_array_type, prefix='docs/example_data/450k_test/g69')
```

This will run a while. It compares all of the probes and clusters CpG probes that show a difference together if they
are close to each other in the genome sequence. There are a lot of adjustable parameters you can play with in
this function. Refer to the docs for more details.

When it completes, it returns a list of files that have been saved to disk:

```
docs/research_notebooks/dmr.acf.txt
docs/research_notebooks/dmr.args.txt
docs/research_notebooks/dmr.fdr.bed.gz
docs/research_notebooks/dmr.manhattan.png
docs/research_notebooks/dmr.regions-p.bed.gz
docs/research_notebooks/dmr.slk.bed.gz
docs/research_notebooks/dmr_regions.csv
docs/research_notebooks/dmr_regions_genes.csv
docs/research_notebooks/dmr_stats.csv
docs/research_notebooks/stats.bed
```

Most of these are intermediate processing files that might be useful when imported into other tools, but the
main summary file is the one that ends in `regions_genes.csv`:

```
chrom  chromStart chromEnd         min_p  ...  z_sidak_p          name      genes distances   descriptions

21     2535657    2535711  6.662000e-12  ...  3.874000e-08  cg06415891  VLDLR-AS1         3   Homo sapiens VLDLR antisense RNA 1 (VLDLR-AS1)...
22     2535842    2535892  2.816000e-04  ...  8.913000e-01  cg12443001      HCG22        16   Homo sapiens HLA complex group 22 (HCG22), ...
25     2577833    2577883  2.048000e-11  ...  1.347000e-07  cg22948959      HLA-C        33   Homo sapiens major histocompatibility compl...
1      876868      876918       0.00303       1.0           cg05475702
1     1514374     1514424      0.003309       1.0           cg00088251
```

This will reveal clusters of CpG probes that were significantly different and annotate these clusters with one or more
nearby genes using the UCSC Genome Browser database. IF you pass in the `tissue=<your tissue type>` argument into
the `diff_meth_regions` function, and that tissue type is one of the 54 that are part of the GTEx (genome expression levels in humans by tissue dataset), this file will also include a column showing the expression levels for any genes
that match, so that you can further narrow down the search for relevant genomic interactions within each experiment.

There are a lot of additional corrections that researchers make at this stage, and many of them are beyond the scope of `methylsuite`, but this function should get you started.

#### gene annotation with UCSC Genome Browser

University of California Santa Cruz maintains a large database of every version of the human genome and its meta data at https://genome.ucsc.edu/cgi-bin/hgTables. You can browse these database tables. IF you are using the latest genome build (hg38), diff_meth_regions will annotate your database using the `refGene` table. It also (partially) supports the `knownGene` and `ncbiRefSeq` tables, if you want to use those. This is useful for identifying genes that are
nearby your regions of interest, and noting the tissue specificity of those genes, in exploring your data.

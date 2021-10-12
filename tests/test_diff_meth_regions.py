import pytest
import methylize
from pathlib import Path
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)

def test_diff_meth_regions_default():

    def run_once():
        import methylcheck
        g69,meta = methylcheck.load_both('/Volumes/LEGX/SCS/GSE69238/')
        meta = meta[ meta.Sample_ID.isin(g69.columns) ] # meta was larger than beta data
        import methylprep
        man_df = methylprep.Manifest('450k').data_frame
        pheno = [1 if x == 'Male' else 0 for x in meta.gender]
        chrom = man_df[man_df.CHR.isin(['16','21'])].index
        sample = g69[ g69.index.isin( chrom ) ]
        sample.to_pickle(Path('tests','test_sample_betas_450k.pkl'))
        import pickle
        with open(Path('tests','test_sample_betas_450k_phenotype.pkl'),'wb') as f:
            pickle.dump(pheno, f)
        print(f"{g69.shape} --> {sample.shape} | pheno: {len(pheno)}")
        return pheno, sample
    #pheno, sample = run_once()

    sample = pd.read_pickle(Path('tests','test_sample_betas_450k.pkl'))
    pheno = pd.read_pickle(Path('tests','test_sample_betas_450k_phenotype.pkl'))
    stats = methylize.diff_meth_pos(sample, pheno, verbose=False)
    manifest_or_array_type = '450k'
    files_created = methylize.diff_meth_regions(stats, manifest_or_array_type, prefix='tests/blah_test/blah_blah')
    print(files_created)
    # BUG: regions file is getting deleted in middle of processing.

    #test_final_results = methylize.fetch_genes('tests/blah_test/blah_blah_stats.csv', save=True)
    #print(test_final_results)

    failures = []
    for _file in files_created:
        if isinstance(_file, type(None)):
            continue
        if not Path(_file).exists():
            failures.append(_file)
    if failures != []:
        raise FileNotFoundError(f"These output files were not found / path missing: {failures}")

def test_diff_meth_positions_no_regions_found():
    # THIS example doesn't return a regions file, because no clusters are found.
    sample = pd.read_pickle(Path('tests','test_sample_betas_epicplus_30k.pkl'))
    pheno = [1,1,0,0,1,1,0,0]
    stats = methylize.diff_meth_pos(sample, pheno, verbose=False)
    manifest_or_array_type = 'epic+'
    files_created = methylize.diff_meth_regions(stats, manifest_or_array_type, prefix='tests/epic_plus/epic_plus')
    print(files_created)

import pytest
import methylize
from pathlib import Path
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)

def test_diff_meth_regions_default():
    test_folder = '450k_test'
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
        with open(Path('docs','example_data','test_sample_betas_450k_phenotype.pkl'),'wb') as f:
            pickle.dump(pheno, f)
        print(f"{g69.shape} --> {sample.shape} | pheno: {len(pheno)}")
        return pheno, sample
    #pheno, sample = run_once()
    sample = pd.read_pickle(Path('docs','example_data','test_sample_betas_450k.pkl'))
    pheno = pd.read_pickle(Path('docs','example_data','test_sample_betas_450k_phenotype.pkl'))
    stats = methylize.diff_meth_pos(sample, pheno, verbose=False)
    manifest_or_array_type = '450k'
    if not Path('docs','example_data', test_folder).exists():
        Path('docs','example_data', test_folder).mkdir()
    files_created = methylize.diff_meth_regions(stats, manifest_or_array_type, prefix='docs/example_data/450k_test/g69')
    print(files_created)
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
    """ also covers:
    [x] passing in manifest
    [x] step None (auto calc)
    [x] verbose True
    [x] genome control True
    cannot test
    - passing in no prefix
    - notebook environment
    """
    test_folder = 'epic_plus'
    if not Path('docs','example_data', test_folder).exists():
        Path('docs','example_data', test_folder).mkdir()
    # THIS example doesn't return a regions file, because no clusters are found.
    sample = pd.read_pickle(Path('docs','example_data','test_sample_betas_epicplus_30k.pkl'))
    pheno = [1,1,0,0,1,1,0,0]
    stats = methylize.diff_meth_pos(sample, pheno, verbose=False)
    #manifest_or_array_type = 'epic+'
    import methylprep
    man = methylprep.Manifest(methlprep.ArrayType('epic+'))
    files_created = methylize.diff_meth_regions(stats, manifest_or_array_type=man, prefix='docs/example_data/epic_plus/epic_plus',
        genome_control=True)
    print(files_created)
    failures = []
    # regions file gets deleted in middle of processing when no regions are found.
    for _file in files_created:
        if isinstance(_file, type(None)):
            continue
        if not Path(_file).exists():
            failures.append(_file)
    if failures != []:
        raise FileNotFoundError(f"These output files were not found / path missing: {failures}")

"""
import pytest
import methylize
from pathlib import Path
import pandas as pd
sample = pd.read_pickle(Path('docs','example_data','test_sample_betas_450k.pkl'))
pheno = pd.read_pickle(Path('docs','example_data','test_sample_betas_450k_phenotype.pkl'))
stats = methylize.diff_meth_pos(sample, pheno, verbose=False)
manifest_or_array_type = '450k'
files_created = methylize.diff_meth_regions(stats, manifest_or_array_type, prefix='docs/example_data/450k_test/g69', tissue='all')
"""

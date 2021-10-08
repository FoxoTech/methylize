import methylprep
import methylcheck
import methylize
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    g69,meta = methylcheck.load_both('/Volumes/LEGX/SCS/GSE69238/')
    meta = meta[ meta.Sample_ID.isin(g69.columns) ] # meta was larger than beta data
    man = methylprep.Manifest('450k')
    man_df = man.data_frame
    g69pheno = [1 if x == 'Male' else 0 for x in meta.gender]
    chromXY = man_df[man_df.CHR.isin(['1','22','23','X'])].index
    g69sample = g69[ g69.index.isin( chromXY ) ]
    print(f"{g69.shape} --> {g69sample.shape} | pheno: {len(g69pheno)}")
    g69stats = methylize.diff_meth_pos(g69sample, g69pheno)
    print('diff_meth_pos done')
    created = methylize.diff_meth_regions(g69stats, man, prefix='dmr', log_to_file=True)
    print(created)

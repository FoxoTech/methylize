import pytest
import methylize as m
from pathlib import Path
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot

#patching
try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch

class TestDMP():
    probes = ['cg20025003', 'cg19303637', 'cg04726019', 'cg24232214', 'cg18722011', 'cg10603385', 'cg14326743', 'cg00614969', 'cg23093450', 'cg10350536', 'cg10606127', 'cg19189310', 'cg27292977', 'cg10280299', 'cg21460394', 'cg20935363', 'cg13436213', 'cg17978996', 'cg18100564', 'cg18687533', 'cg01754290', 'cg06815817', 'cg11013355', 'cg08006166', 'cg12444081', 'cg04169610', 'cg03002586', 'cg12525736', 'cg07221080', 'cg14645926', 'cg24140204', 'cg04097334', 'cg15122841', 'cg12984107', 'cg26721264', 'cg18250453', 'cg07356722', 'cg15917060', 'cg04329454', 'cg25915536', 'cg23819144', 'cg03960811', 'cg15122372', 'cg02565891', 'cg26858414', 'cg07551659', 'cg04690110', 'cg01119145', 'cg07127883', 'cg13883671', 'cg04422084', 'cg24208375', 'cg04720592', 'cg11557307', 'cg01659631', 'cg04891053', 'cg25690589', 'cg27151629', 'cg12046564', 'cg09286880', 'cg13507658', 'cg03526902', 'cg23865597', 'cg16726760', 'cg00686643', 'cg06932713', 'cg15632164', 'cg27434487', 'cg04984709', 'cg18785833', 'cg02390188', 'cg16279999', 'cg06398335', 'cg21996139', 'cg10186768', 'cg00252032', 'cg02752267', 'cg25838590', 'cg27585074', 'cg26976732', 'cg10028549', 'cg26540370', 'cg01710666', 'cg17585066', 'cg13654376', 'cg20731167', 'cg20849032', 'cg22547851', 'cg20643159', 'cg16255631', 'cg03779097', 'cg11860160', 'cg23505303', 'cg08770734', 'cg10767425', 'cg15335117', 'cg11302624', 'cg06999856', 'cg09533178', 'cg22428541', 'cg19589795', 'cg11598658', 'cg25020142', 'cg19139092', 'cg13067974', 'cg20659418', 'cg13474011', 'cg05751839', 'cg02907689', 'cg00584026', 'cg14013604', 'cg06291428', 'cg09925855', 'cg27032056', 'cg22851252', 'cg15055577', 'cg02563156', 'cg11626052', 'cg20583432', 'cg00871578', 'cg00187465', 'cg05684665', 'cg19734374', 'cg26522792', 'cg17333886', 'cg07353572', 'cg01622668', 'cg23601030', 'cg00941824', 'cg07863831', 'cg10241005', 'cg24938560', 'cg03125090', 'cg17587682', 'cg12967902', 'cg02980621', 'cg07609108', 'cg13682019', 'cg21668803', 'cg10183150', 'cg14367014', 'cg26540302', 'cg10057843', 'cg10884539', 'cg25134747', 'cg24376537', 'cg04993290', 'cg06073417', 'cg01974656', 'cg18175470', 'cg17364949', 'cg13798270', 'cg06869505', 'cg06300223', 'cg23888902', 'cg22549986', 'cg10888111', 'cg07693813', 'cg13894822', 'cg05134775', 'cg05041061', 'cg25223567', 'cg12806067', 'cg20000541', 'cg21472953', 'cg24759237', 'cg21388745', 'cg03901777', 'cg10135532', 'cg00708060', 'cg03515098', 'cg19328682', 'cg23165164', 'cg10098787', 'cg11466449', 'cg07740071', 'cg19375583', 'cg01121127', 'cg12260798', 'cg01251854', 'cg18098567', 'cg00514284', 'cg14969789', 'cg13216327', 'cg05071422', 'cg17873456', 'cg14372532', 'cg04260065', 'cg12151820', 'cg11972796', 'cg14298200', 'cg13459764', 'cg20948024', 'cg22860891', 'cg08962590', 'cg23244022', 'cg17173663', 'cg04722610', 'cg21703322', 'cg04841358']
    test_beta = pd.read_pickle(Path('data/GSE69852_beta_values.pkl')).loc[probes]
    test_m = pd.read_pickle(Path('data/GSE69852_m_values.pkl')).loc[probes]
    test_mouse = pd.read_pickle(Path('data/mouse_beta_values.pkl'))
    test_mouse = test_mouse.mask( test_mouse.le(0.98) & test_mouse.ge(0.02) ).dropna().sample(1500) # mask yields 7224 probes
    meta = pd.read_pickle('data/GSE69852_GPL13534_meta_data.pkl')
    # use source or converted_age (took strings of years or weeks and made float on same time scale)

    @patch("matplotlib.pyplot.show")
    def test_dmp_linear_age_450k(self, mock):
        res = m.diff_meth_pos(self.test_m, self.meta['converted_age'], 'linear', verbose=True)
        if res.shape != (200,7):
            raise AssertionError(f"results shape wrong {res.shape} vs (200,7)")
        m.manhattan_plot(res, '450k')
        if res.shape != (200,10):
            raise AssertionError(f"results shape wrong {res.shape} vs (200,10)")

    @patch("matplotlib.pyplot.show")
    def test_dmp_linear_age_450k_debug(self, mock):
        filename = Path('data/test_lin.png')
        res = m.diff_meth_pos(self.test_m, self.meta['converted_age'], 'linear', debug=True)
        if res.shape != (200,7):
            raise AssertionError(f"results shape wrong {res.shape} vs (200,7)")
        m.manhattan_plot(res, '450k', no_thresholds=True)
        print('no_thresholds OK')
        m.manhattan_plot(res, '450k', explore=True)
        print('explore OK')
        m.manhattan_plot(res, '450k', plain=True)
        print('plain OK')
        m.manhattan_plot(res, '450k', statsmode=True, plot_cutoff_label=False, label_sig_probes=False)
        print('no labels OK')
        m.manhattan_plot(res, '450k', fdr=False, bonferroni=False, suggestive=1e-3, significant=1e-7, fwer=0.001)
        print('custom lines OK')
        m.manhattan_plot(res, '450k', plain=True, genome_build='OLD', label_prefix='CHR-', save=True, filename=filename)
        if filename.exists():
            print(f'saved OK: {filename}')
            filename.unlink()
        m.manhattan_plot(res, '450k', fdr=False, bonferroni=False, suggestive=1e-3, significant=1e-7, fwer=0.001, save=True, filename=filename)
        if filename.exists():
            print(f'saved OK: {filename}')
            filename.unlink()
        m.volcano_plot(res, save=True, cutoff=(-0.05,0.05), filename=filename)
        if filename.exists():
            print(f'saved volcano OK: {filename}')
            filename.unlink()

    @patch("matplotlib.pyplot.show")
    def test_dmp_logistic_treatment_450k(self, mock):
        #test_m = pd.read_pickle(Path('data/GSE69852_m_values.pkl')).sample(1000)
        res = m.diff_meth_pos(self.test_m, self.meta['source'], 'logistic')
        m.manhattan_plot(res, '450k')
        ### NOTE: (88, 10) is result locally, but (89, 10) is the result on circleci / github-actions
        if res.shape not in [(89,10), (88,10)]:
            raise AssertionError(f"results shape wrong {res.shape} vs [(89, 10),(88,10)]")
        m.volcano_plot(res, alpha=0.05, adjust=True, fwer=0.01, width=20, height=20, fontsize=14, dotsize=99)
        filename = Path('data/test_logres.png')
        m.volcano_plot(res, alpha=0.05, adjust=False, fwer=0.4, width=20, height=20, fontsize=14, dotsize=99, save=True, filename=filename)
        if filename.exists():
            print(f'saved volcano OK: {filename}')
            filename.unlink()
        res = m.diff_meth_pos(self.test_m, self.meta['source'], 'logistic', verbose=True, debug=True)
        if res.shape not in [(89,7),(88,7)]: # 7 before plot; 10 after plot
            raise AssertionError(f"results shape wrong {res.shape} vs [(89,7),(88,7)]")
        ref = {'Coefficient': -36.19092004319955, 'PValue': 0.5952467603296255,
            'StandardError': 12626.740574121952, 'fold_change': 0.053637267861368014,
            '95%CI_lower': -1.2397613636363634, '95%CI_upper': -0.5005227272727276,
            'FDR_QValue': 0.9972779212722235}
        checks_out = np.isclose( pd.Series(dict(res.mean())), pd.Series(ref) ).all()
        if checks_out is False:
            raise AssertionError("Mean values for results DF returned do not match reference values:\n({dict(res.mean())} vs ref {ref})")

    @patch("matplotlib.pyplot.show")
    def test_dmp_logistic_with_covariates(self, mock):
        pheno = pd.DataFrame(data={
            'gender': ['Male','Male','Female','Female','Female','Male'],
            'age': [5,24,36,5,52,30],
            'treatment': ['healthy','diseased','healthy','diseased','healthy','diseased']
        })
        res = m.diff_meth_pos(self.test_mouse, pheno, debug=True, column='treatment', covariates=True)
        #pheno['dummy'] = 5.0 # testing zero variance case, and creating many Linear Algebra errors.
        pheno['dummy'] = np.random.choice([1, 3, 5, 6,9], pheno.shape[0]) # prevents high freq of LinAlg errors
        res4 = m.diff_meth_pos(self.test_mouse, pheno, debug=True, column='treatment', covariates=True)
        m.manhattan_plot(res, 'mouse', save=True, filename='data/test-res.png')
        m.manhattan_plot(res4, 'mouse', save=True, filename='data/test-res4.png')
        m.volcano_plot(res)
        if Path('data/test-res.png').exists():
            print(Path('data/test-res.png').stat().st_size)
            Path('data/test-res.png').unlink()
        else:
            raise Exception("manhattan plot file not found")
        if Path('data/test-res4.png').exists():
            print(Path('data/test-res4.png').stat().st_size)
            Path('data/test-res4.png').unlink()
        else:
            raise Exception("manhattan plot file not found")


"""
def test():
    import pandas as pd
    import methylize
    p64 = pd.read_pickle('Project_064_test/beta_values.pkl')
    p64meta = [1,1,0,0,1,1,0,0]
    stats = methylize.diff_meth_pos(p64.sample(100000), p64meta)
    print(f"stats; sig probes: {(stats.FDR_QValue < 0.05).sum()} | {(stats.PValue < 0.05).sum()}")
    methylize.manhattan_plot(stats, 'epic+')
    return stats

def test(adjust=True):
    # run in /Volumes/LEGX/GEO/GSE143411
    import pandas as pd
    import methylize
    import random
    import methylcheck
    df = methylcheck.load('.')
    pheno = [random.choice([0,1]) for i in range(len(df.columns))]
    stats = methylize.diff_meth_pos(df.sample(60000), pheno)
    print(f"stats; sig probes: {(stats.FDR_QValue < 0.05).sum()} | {(stats.PValue < 0.05).sum()}")
    #methylize.manhattan_plot(stats, '450k', adjust=adjust)
    methylize.volcano_plot(stats, plot_cutoff_label=True)
    return stats

def testage():
    folder = '/Volumes/LEGX/GEO/GSE85566/GPL13534'
    import pandas as pd
    import methylize
    import random
    import methylcheck
    df = methylcheck.load(folder)
    meta = pd.read_pickle('/Volumes/LEGX/GEO/GSE85566/GPL13534/GSE85566_GPL13534_meta_data.pkl')
    pheno = meta.age
    print(df.shape, len(meta.age))
    stats = methylize.diff_meth_pos(df.sample(60000), pheno, regression_method='linear', fwer=0.05)
    #methylize.manhattan_plot(stats, '450k', adjust=True)
    methylize.volcano_plot(stats, cutoff='auto')
    methylize.volcano_plot(stats, cutoff=None)
    return stats

# see line 643 -- where p 0.05 is too high
# line 292, 561  -- log-regress faulty

    # run in /Volumes/LEGX/GEO/GSE85566/GPL13534
    #pheno = meta.ethnicity # .gender was overproducing differences. CAN ONLY HAVE 2 categories
    #stats = methylize.diff_meth_pos(df.sample(60000), pheno, regression_method='logistic', fwer=0.1)
    #print(f"stats; sig probes: {(stats.FDR_QValue < 0.05).sum()}")
    #methylize.manhattan_plot(stats, '450k')
    #methylize.volcano_plot(stats, plot_cutoff_label=True, beta_coefficient_cutoff=(-0.02, 0.02), cutoff=0.05)

def test():
    folder = '/Volumes/LEGX/GEO/GSE85566/GPL13534'
    import pandas as pd
    import methylize
    import random
    import methylcheck
    df = pd.read_csv('test_probes.csv').set_index('Unnamed: 0')
    pheno = pd.read_csv('/Volumes/LEGX/GEO/GSE85566/GPL13534/phenotypes.csv')['0']
    stats = methylize.diff_meth_pos(df, pheno, regression_method='logistic', fwer=0.05)
    methylize.volcano_plot(stats)
    #stats = methylize.diff_meth_pos(df, pheno, regression_method='logistic', fwer=0.05, scratch=True)
    methylize.volcano_plot(stats, cutoff='auto')
    return stats

def test():
    folder = '/Volumes/LEGX/GEO/GSE85566/GPL13534'
    import pandas as pd
    import methylize
    import random
    import methylcheck
    df = methylcheck.load(folder)
    meta = pd.read_pickle('/Volumes/LEGX/GEO/GSE85566/GPL13534/GSE85566_GPL13534_meta_data.pkl')
    pheno = meta.ethnicity.replace({'Other': 'EA'})
    print(df.shape, len(meta.ethnicity))
    #pheno = meta['disease status']
    stats = methylize.diff_meth_pos(df.sample(30000), pheno, regression_method='logistic', fwer=0.05)
    methylize.volcano_plot(stats, plot_cutoff_label=True, beta_coefficient_cutoff=(-0.2, 0.2), adjust=None, cutoff=0.05, fwer=0.05)
    return stats


def abh(pvals, q=0.05): # another false discovery rate from scratch method
    pvals[pvals>0.99] = 0.99 # P-values equal to 1. will cause a division by zero.
    def lsu(pvals, q=0.05):
        m = len(pvals)
        sort_ind = np.argsort(pvals)
        k = [i for i, p in enumerate(pvals[sort_ind]) if p < (i+1.)*q/m]
        significant = np.zeros(m, dtype='bool')
        if k:
            significant[sort_ind[0:k[-1]+1]] = True
        return significant
    significant = lsu(pvals, q) # If lsu does not reject any hypotheses, stop
    if significant.all() is False:
        return significant
    m = len(pvals)
    sort_ind = np.argsort(pvals)
    m0k = [(m+1-(k+1))/(1-p) for k, p in enumerate(pvals[sort_ind])]
    j = [i for i, k in enumerate(m0k[1:]) if k > m0k[i-1]]
    mhat0 = int(np.ceil(min(m0k[j[0]+1], m)))
    qstar = q*m/mhat0
    return lsu(pvals, qstar)

def fdr(p_vals):
    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr

def junk
    #data = methylcheck.load(path, format='beta_csv')
    #data = pd.read_pickle('/Volumes/LEGX/GEO/test_pipeline/GSE111629/beta_values.pkl')
    meta.source = meta.source.str.replace('X','')
    meta = meta[meta.source.isin(data.columns)]
    pheno = list(meta['disease state']) # [:-1] # off by one with sample data for full datasets

def man2():
    import methylize as m
    import pandas as pd
    from pathlib import Path
    path = Path('/Volumes/LEGX/GEO/GSE168921/')
    meta = pd.read_pickle(Path(path, 'sample_sheet_meta_data.pkl'))
    data = pd.read_pickle(Path(path, 'beta_values.pkl'))
    pheno = list(meta.sample_group) # --- scratch requires a list, not a series
    sample = data.sample(150000);print(sample)
    res = m.diff_meth_pos(sample, pheno, 'logistic', export=False, impute='average', debug=True)
    #m.manhattan_plot(res, '450k', fontsize=10, save=False, palette='Gray')
    return res

def mantest():
    import methylize as m
    import pandas as pd
    from pathlib import Path
    path = Path('/Volumes/LEGX/GEO/GSE168921/')
    meta = pd.read_pickle(Path(path, 'sample_sheet_meta_data.pkl'))
    data = pd.read_pickle(Path(path, 'beta_values.pkl'))
    pheno = meta.sample_group
    sample = data.sample(150000);print(sample)
    res = m.diff_meth_pos(sample, pheno, 'logistic', export=False, impute='average')
    #m.manhattan_plot(res, '450k', fontsize=10, save=False, palette='Gray')
    return res

def voltest():
    import methylize as m
    import pandas as pd
    meth_data = pd.read_pickle('data/GSE69852_beta_values.pkl').transpose()
    pheno_data = ["0","33","0","52","0","57"]
    res = m.diff_meth_pos(meth_data.sample(15000,axis=1), pheno_data, 'linear', export=False)
    m.volcano_plot(res, adjust=True)

def logistest():
    from random import random
    import methylize as m
    import pandas as pd
    from pathlib import Path
    path = Path('/Volumes/LEGX/GEO/GSE85566/')
    df1 = pd.read_csv(Path(path,'beta_dropped_test.csv')).set_index('IlmnID')
    df2 = pd.read_csv(Path(path,'beta_imputed_test.csv')).set_index('IlmnID')
    pheno = pd.read_pickle(Path(path,'GPL13534','GSE85566_GPL13534_meta_data.pkl'))
    pheno_alt = pheno[pheno.ethnicity != 'Other']
    pheno_vector = pheno_alt.ethnicity
    # drop samples for 'Other' ethnicity
    df1_alt = df1[pheno_alt.Sample_ID]
    df2_alt = df2[pheno_alt.Sample_ID]
    # replace some probes
    probes = ['cg00206063', 'cg00328720', 'cg00579868', 'cg00664723', 'cg00712106']
    ref = pheno_alt[['Sample_ID','ethnicity']].set_index('Sample_ID')
    row = [0.01 + random()/1000 if v == 'AA' else 0.99 - random()/1000 for v in ref.values]
    for probe in probes:
        print(len(row), df1_alt.shape, df2_alt.shape, pheno_alt.shape)
        df1_alt.loc[probe] = row
        df2_alt.loc[probe] = row
    print(df1_alt.head())
    # convert to M-values
    import math
    def beta2m(val):
        return math.log2(val/(1-val))
    df1_alt = df1_alt.applymap(beta2m)
    print(df1_alt.head())
    result1 = m.diff_meth_pos(df1_alt, pheno_vector, 'logistic', export=False, verbose=True)
    result2 = m.diff_meth_pos(df2_alt, pheno_vector, 'logistic', export=False, verbose=True)
    return result1, result2

"""

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
    test_mouse = Path('data/mouse_beta_values.pkl')
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
        if res.shape != (88,10):
            raise AssertionError(f"results shape wrong {res.shape} vs (88,10)")
        m.volcano_plot(res, alpha=0.05, adjust=True, fwer=0.01, width=20, height=20, fontsize=14, dotsize=99)
        filename = Path('data/test_logres.png')
        m.volcano_plot(res, alpha=0.05, adjust=False, fwer=0.4, width=20, height=20, fontsize=14, dotsize=99, save=True, filename=filename)
        if filename.exists():
            print(f'saved volcano OK: {filename}')
            filename.unlink()
        res = m.diff_meth_pos(self.test_m, self.meta['source'], 'logistic', verbose=True, debug=True)
        if res.shape != (88, 7): # 7 before plot; 10 after plot
            raise AssertionError(f"results shape wrong {res.shape} vs (88,7)")
        ref = {'Coefficient': -36.19092004319955, 'PValue': 0.5952467603296255,
            'StandardError': 12626.740574121952, 'fold_change': 0.053637267861368014,
            '95%CI_lower': -1.2397613636363634, '95%CI_upper': -0.5005227272727276,
            'FDR_QValue': 0.9972779212722235}
        checks_out = np.isclose( pd.Series(dict(res.mean())), pd.Series(ref) ).all()
        if checks_out is False:
            raise AssertionError("Mean values for results DF returned do not match reference values:\n({dict(res.mean())} vs ref {ref})")

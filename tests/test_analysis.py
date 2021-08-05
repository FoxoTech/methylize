import pandas as pd
#patching
#try:
#    # python 3.4+ should use builtin unittest.mock not mock package
#    from unittest.mock import patch
#except ImportError:
#    from mock import patch
import methylize

meth_data = pd.read_pickle('data/GSE69852_beta_values.pkl').transpose()

class TestInit():

    def test_is_interactive(self):
        from methylize.diff_meth_pos import is_interactive
        if is_interactive() == False:
            pass
        else:
            raise ValueError()

    def test_diff_meth_pos_logistic(self):
        pheno_data = ["fetal","fetal","fetal","adult","adult","adult"]
        test_results = methylize.diff_meth_pos(
            meth_data.sample(1000, axis=1),
            pheno_data,
            regression_method="logistic",
            export=False,
            max_workers=2)
        return test_results

    def test_diff_meth_pos_linear(self):
        pheno_data = ["0","0","0","52","54","57"]
        test_results = methylize.diff_meth_pos(
            meth_data.sample(1000, axis=1),
            pheno_data,
            regression_method="linear",
            export=False,
            max_workers=2)
        return test_results

    def test_manhattan(self):
        test_results= self.test_diff_meth_pos_logistic()
        methylize.manhattan_plot(test_results, '450k', cutoff=0.01, palette='Gray3', save=False)

    def test_volcano(self):
        test_results= self.test_diff_meth_pos_linear()
        methylize.volcano_plot(test_results, fontsize=10, cutoff=0.15, beta_coefficient_cutoff=(-0.05,0.05), save=False)

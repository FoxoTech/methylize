import pandas as pd
import matplotlib.pyplot
import unittest
#patching
try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch
import methylize

meth_data = pd.read_pickle('data/GSE69852_beta_values.pkl').transpose()

class TestInit():


    #@patch("matplotlib.pyplot.show")
    def test_all_manifests_load_with_genomic_info(self): #, mock):
        """Also tests all probe2chr mapping options, for all arrays."""
        import methylprep
        expected_columns = list(methylprep.files.manifests.MANIFEST_COLUMNS) + ['probe_type']
        expected_columns.remove('IlmnID')
        mouse_columns = list(methylprep.files.manifests.MOUSE_MANIFEST_COLUMNS) + ['probe_type']
        mouse_columns.remove('IlmnID')
        for this_array_type in ['450k', 'epic', 'epic+']:
            man = methylprep.Manifest(methylprep.ArrayType(this_array_type))
            found = set(man.data_frame.columns)
            if found != set(expected_columns):
                raise AssertionError("manifest columns don't match expected columns, defined in methylprep.files.manifests.py")
            probe_map = methylize.helpers.create_probe_chr_map(man, 'CHR', include_sex=True)
            df = pd.DataFrame.from_dict(probe_map, orient='index')
            print(f"{this_array_type} CHR include_sex")
            print(df.value_counts())
            probe_map = methylize.helpers.create_probe_chr_map(man, 'OLD_CHR', include_sex=False)
            df = pd.DataFrame.from_dict(probe_map, orient='index')
            print(f"{this_array_type} OLD_CHR exclude_sex")
            print(df.value_counts())
        man = methylprep.Manifest(methylprep.ArrayType('mouse'))
        found = set(man.data_frame.columns)
        if found != set(mouse_columns):
            raise AssertionError("manifest columns don't match expected columns, defined in methylprep.files.manifests.py")
        probe_map = methylize.helpers.create_probe_chr_map(man, 'CHR', include_sex=False)
        df = pd.DataFrame.from_dict(probe_map, orient='index')
        print(f"{this_array_type} CHR exclude_sex")
        print(df.value_counts())
        probe_map = methylize.helpers.create_probe_chr_map(man, 'OLD_CHR', include_sex=True)
        df = pd.DataFrame.from_dict(probe_map, orient='index')
        print(f"{this_array_type} OLD_CHR include_sex")
        print(df.value_counts())


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

    @patch("matplotlib.pyplot.show")
    def test_manhattan(self, mock):
        test_results= self.test_diff_meth_pos_logistic()
        methylize.manhattan_plot(test_results, '450k') #, cutoff=0.01, palette='Gray3', save=False)

    @patch("matplotlib.pyplot.show")
    def test_volcano(self, mock):
        test_results= self.test_diff_meth_pos_linear()
        methylize.volcano_plot(test_results, fontsize=10, cutoff=0.15, beta_coefficient_cutoff=(-0.05,0.05), save=False)

    @patch("matplotlib.pyplot.show")
    def test_diff_meth_pos_mouse_linear(self, mock):
        """mouse contains NaNs and multiple probes of the same IlmnID"""
        ms = pd.read_pickle('data/mouse_beta_values.pkl')
        pheno_data = [3,5,7,9,11,13]
        test_results = methylize.diff_meth_pos(ms.transpose().sample(10000,axis=1), pheno_data, regression_method='linear', impute=True)
        if len(test_results) == 0: # happens with logistic method + mouse
            raise AssertionError("no test_results for diff_meth_pos() on mouse")
        methylize.manhattan_plot(test_results, 'mouse', save=False)
        methylize.volcano_plot(test_results, save=False)

    @patch("matplotlib.pyplot.show")
    def test_diff_meth_pos_mouse_logistic(self, mock):
        """mouse contains NaNs and multiple probes of the same IlmnID"""
        impute_types = ['delete', 'fast', True, 'auto', 'average']
        ms = pd.read_pickle('data/mouse_beta_values.pkl')
        pheno_data = ['a','a','a','b','b','b']
        for impute_type in impute_types:
            print(f"***{impute_type}***")
            test_results = methylize.diff_meth_pos(ms.transpose().sample(2000,axis=1), pheno_data, regression_method='logistic', impute=impute_type)
            if len(test_results) == 0: # happens with logistic method + mouse
                raise AssertionError("no test_results for diff_meth_pos() on mouse")
            methylize.manhattan_plot(test_results, 'mouse', save=False)
            methylize.volcano_plot(test_results, save=False)


class ErrorTest(unittest.TestCase):
    @patch("matplotlib.pyplot.show")
    def test_diff_meth_pos_mouse_logistic_throws_errors(self, mock):
        """mouse contains NaNs and multiple probes of the same IlmnID"""
        ms = pd.read_pickle('data/mouse_beta_values.pkl')
        pheno_data = [3,5,7,9,11,13]
        # too many phenotypes for logistic (must be exactly 2)
        with self.assertRaises(ValueError):
            methylize.diff_meth_pos(ms.transpose().sample(1000,axis=1), pheno_data, regression_method='logistic', impute='fast')
        # NaNs and impute=False
        with self.assertRaises(ValueError):
            methylize.diff_meth_pos(ms.transpose().sample(1000,axis=1), pheno_data, regression_method='logistic', impute=False)
        # passing some other kwargs
        with self.assertRaises(ValueError):
            methylize.diff_meth_pos(ms.transpose().sample(1000,axis=1), pheno_data, regression_method='logistic', impute='blah')


'''
def test13():
    import methylize
    import pandas as pd
    ms = pd.read_pickle('data/mouse_beta_values.pkl')
    pheno_data = ['a','b','a','a','b','b']
    test_results = methylize.diff_meth_pos(ms.transpose().sample(10000,axis=1), pheno_data, regression_method='logistic', impute='fast')

def test12():
    import methylize
    import pandas as pd
    ms = pd.read_pickle('data/mouse_beta_values.pkl')
    pheno_data = [3,5,7,9,11,13]
    test_results = methylize.diff_meth_pos(ms.transpose().sample(1000,axis=1), pheno_data, regression_method='linear')

def test11():
    import methylize
    import pandas as pd
    import random
    data = pd.read_pickle('/Volumes/LEGX/stp1/saliva_qc_pass.pkl')
    pheno_data = [random.randint(0,1) for i in range(data.shape[0])]
    print(len(pheno_data), pheno_data)
    print('converting to float32 first')
    data = data.astype('float32')
    return methylize.diff_meth_pos(data.sample(50000,axis=1), pheno_data, regression_method='logistic', impute='delete')

def test10():
    import methylize
    import pandas as pd
    pheno_data = [1,2,3,4,5,6,7,8,9,10,11,12]
    data = pd.read_pickle('/Volumes/LEGX/55085/beta_values.pkl')
    return methylize.diff_meth_pos(data.transpose().sample(50000,axis=1), pheno_data, regression_method='linear', impute=True)

def test9():
    import methylize
    import pandas as pd
    pheno_data = ["fetal","adult","fetal","adult","fetal","adult","fetal","adult","fetal","adult","fetal","fetal"]
    data = pd.read_pickle('/Volumes/LEGX/55085/beta_values.pkl')
    return methylize.diff_meth_pos(data.transpose().sample(50000,axis=1), pheno_data, regression_method='logistic', impute=True)

def test8():
    import methylize
    import pandas as pd
    pheno_data = ["fetal","adult","fetal","adult","fetal","adult","fetal","adult","fetal","adult","fetal","adult","fetal","adult","fetal","adult","fetal","adult","fetal"]
    data = pd.read_pickle('/Volumes/LEGX/GEO/GSE168808/GPL21145/beta_values.pkl')
    return methylize.diff_meth_pos(data.transpose().sample(50000,axis=1), pheno_data, regression_method='logistic', impute=True)

def test7():
    import methylize
    import pandas as pd
    pheno_data = ["fetal","adult"]
    data = pd.read_pickle('/Volumes/LEGX/GEO/GSE168808/GPL13534/beta_values.pkl')
    return methylize.diff_meth_pos(data.transpose().sample(50000,axis=1), pheno_data, regression_method='logistic', impute=True)

def test5():
    import methylize
    import pandas as pd
    pheno_data = ["fetal","fetal","fetal","adult","adult","adult","adult","adult","adult","fetal","fetal","fetal"]
    ms = pd.read_pickle('beta_values.pkl')
    return methylize.diff_meth_pos(ms.transpose().sample(50000,axis=1), pheno_data, regression_method='logistic', impute=True)

def test6():
    import methylize
    import pandas as pd
    pheno_data = [1,2,3,4,6,6,10,8,9,10,12,12]
    ms = pd.read_pickle('beta_values.pkl')
    return methylize.diff_meth_pos(ms.transpose().sample(50000,axis=1), pheno_data, regression_method='linear', impute=True)

def test1():
    import methylize
    import pandas as pd
    pheno_data = ["fetal","fetal","fetal","adult","adult","adult"]
    #ms = pd.read_pickle('mouse_beta_values.pkl')
    ms = pd.read_pickle('~/methylize/data/mouse_beta_values.pkl')
    return methylize.diff_meth_pos(ms.transpose().sample(50000,axis=1), pheno_data, regression_method='logistic', impute=True)

def test2():
    import methylize
    import pandas as pd
    pheno_data = ["fetal","fetal","fetal","adult","adult","adult"]
    ms = pd.read_pickle('~/methylize/data/GSE69852_beta_values.pkl')
    return methylize.diff_meth_pos(ms.transpose().sample(50000,axis=1), pheno_data, regression_method='logistic', impute=True)

def test3():
    import methylize
    import pandas as pd
    pheno_data = [3,5,7,9,11,13]
    ms = pd.read_pickle('mouse_beta_values.pkl')
    return methylize.diff_meth_pos(ms.transpose().sample(50000,axis=1), pheno_data, regression_method='linear', impute=True)

def test4(): # this never returns any stats
    import methylize
    import pandas as pd
    pheno_data = ['bob','sue','bob','sue','bob','sue']
    ms = pd.read_pickle('mouse_beta_values.pkl')
    return methylize.diff_meth_pos(ms.transpose().sample(50000,axis=1), pheno_data, regression_method='logistic', impute=True)

import methylize
import pandas as pd
ms = pd.read_pickle('data/mouse_beta_values.pkl')
pheno_data = [3,5,7,9,11,13]
test_results = methylize.diff_meth_pos(ms, pheno_data)
methylize.manhattan_plot(test_results, 'mouse')

#ms.transpose().sample(10000,axis=1), pheno_data, regression_method='linear', impute=True)
'''

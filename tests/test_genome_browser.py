import pytest
import pandas as pd
from pathlib import Path
import methylize

# tests
"""
- test raw sql
- pull and test with every kind of ref database
- look at number of identified genes returned in each dataframe
change tol
without cache
verbose ON
save on/off
cannot test "flush cache after 30 days" code
"""

class TestGenome():
    source = Path('tests','test_dmr_regions.csv')
    expected_match_tol_250 = (1153, 11)
    expected_match_tol_100 = (499, 65) # tissue='all'
    expected_match_tol_10 = (56, 12) # tissue='blood'
    expected_match_tol_250_ncbi = (1626, 11)
    expected_match_tol_100_ncbi = (760, 11)
    expected_match_tol_250_known = (2605, 11) # (2637, 11) -- when using latest copy; (2605, 11) --- using cached
    expected_match_tol_100_known = (1350, 11)

    def test_fetch_genes_sql(self):
        sql = """SELECT name, name2, txStart, txEnd, description FROM refGene LEFT JOIN kgXref ON kgXref.refseq = refGene.name limit 10;"""
        results = methylize.fetch_genes(self.source, sql=sql)
        if len(results) != 10:
            raise AssertionError(f"raw SQL option failed")

    def test_fetch_genes_ncbiRefSeq(self):
        results = methylize.fetch_genes(self.source, ref='ncbiRefSeq',  no_sync=True, tol=250)
        # no descriptions (can't join tables)
        matched = results[ results.genes != '' ]
        if matched.shape != self.expected_match_tol_250_ncbi:
            raise AssertionError(f"fetch_genes matched {matched.shape}; expected {self.expected_match_tol_250_ncbi}. Perhaps the genome data updated?")

    def test_fetch_genes_knownGene(self):
        results = methylize.fetch_genes(self.source, ref='knownGene', tol=250, use_cached=True, no_sync=True)
        matched = results[ results.genes != '' ]
        if matched.shape != self.expected_match_tol_250_known:
            raise AssertionError(f"fetch_genes matched {matched.shape}; expected {self.expected_match_tol_250_known}. Perhaps the genome data updated?")

    def test_fetch_genes_refGene(self):
        """ was flaky and slow on github actions, but merely slow on circleci, so disabling for now """
        results = methylize.fetch_genes(self.source, ref='refGene', tol=10, tissue='blood', use_cached=True, no_sync=True)
        matched = results[ results.descriptions != '' ]
        if matched.shape != self.expected_match_tol_10:
            raise AssertionError(f"fetch_genes matched {matched.shape}; expected {self.expected_match_tol_10}. Perhaps the genome data updated?")

    def test_fetch_genes_errors(self):
        """ does not test downloading the data from UCSC"""
        with pytest.raises(Exception) as e_msg:
            """either one or the other"""
            results = methylize.fetch_genes(None, sql=None, verbose=True)
            assert str(e_msg.value) == "Either provide a path to the DMR stats file or a sql query."
        with pytest.raises(Exception) as e_msg:
            import pandas as pd
            fake_df = pd.DataFrame(data={'name':['a','b','c'],'chromEnd': [5,26,79]})
            fake_source = Path('tests','fake_regions.csv')
            fake_df.to_csv(fake_source)
            results = methylize.fetch_genes(fake_source, verbose=True)
        print(str(e_msg.value))
        Path(fake_source).unlink()

    if Path('tests','test_dmr_regions_genes.csv').exists():
        Path('tests','test_dmr_regions_genes.csv').unlink()

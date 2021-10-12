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
    expected_match_tol_100 = (499, 11)
    expected_match_tol_100_ncbi = (760, 11)
    expected_match_tol_100_known = (1350, 11)

    def test_fetch_genes_sql(self):
        sql = """SELECT name, name2, txStart, txEnd, description FROM refGene LEFT JOIN kgXref ON kgXref.refseq = refGene.name limit 10;"""
        results = methylize.fetch_genes(self.source, sql=sql)
        if len(results) != 10:
            raise AssertionError(f"raw SQL option failed")

    def test_fetch_genes_ncbiRefSeq(self):
        results = methylize.fetch_genes(self.source, ref='ncbiRefSeq')
        # no descriptions (can't join tables)
        matched = results[ results.genes != '' ]
        if matched.shape != self.expected_match_tol_100_ncbi:
            raise AssertionError(f"fetch_genes matched {matched.shape}; expected {self.expected_match_tol_100_ncbi}. Perhaps the genome data updated?")

    def test_fetch_genes_knownGene(self):
        results = methylize.fetch_genes(self.source, ref='knownGene')
        matched = results[ results.genes != '' ]
        if matched.shape != self.expected_match_tol_100_known:
            raise AssertionError(f"fetch_genes matched {matched.shape}; expected {self.expected_match_tol_100}. Perhaps the genome data updated?")

    def test_fetch_genes_refGene(self):
        results = methylize.fetch_genes(self.source, ref='refGene')
        matched = results[ results.descriptions != '' ]
        if matched.shape != self.expected_match_tol_100:
            raise AssertionError(f"fetch_genes matched {matched.shape}; expected {self.expected_match_tol_100}. Perhaps the genome data updated?")

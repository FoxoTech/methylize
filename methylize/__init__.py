import logging
from .diff_meth_pos import diff_meth_pos, volcano_plot, manhattan_plot
from .diff_meth_regions import diff_meth_regions
from .genome_browser import fetch_genes
from .helpers import to_BED
from .version import __version__
from . import cpv

logging.basicConfig(level=logging.INFO)

__all__ = [
    'diff_meth_pos',
    'volcano_plot',
    'manhattan_plot',
    'diff_meth_regions',
    'fetch_genes',
    'to_BED',
    'cpv',
]

import logging
from .diff_meth_pos import diff_meth_pos, volcano_plot, manhattan_plot
from .diff_meth_regions import diff_meth_regions
from .helpers import to_genome, to_BED
from .version import __version__

logging.basicConfig(level=logging.INFO)

__all__ = [
    'diff_meth_pos', 'volcano_plot', 'manhattan_plot', 'diff_meth_regions',
]

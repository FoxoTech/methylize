import logging
from .diff_meth_pos import diff_meth_pos, volcano_plot, manhattan_plot

logging.basicConfig(level=logging.INFO)

__all__ = ['diff_meth_pos', 'volcano_plot', 'manhattan_plot']

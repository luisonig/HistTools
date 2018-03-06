#! /usr/bin/env python

import os
import sys
import re
import gzip
#import glob

from highisttools import *

class InParams:
    DEFAULT_ACTION = 'print_xs'

    def __init__(self):
        from argparse import ArgumentParser, FileType
        parser = ArgumentParser(description='Plots histogram "histo"')
    
        parser.add_argument('-w', '--overwrite', action='store_true', dest="OVERWRITE",
                            default=False, help="whether to overwrite an existing folder [NO].")
        
        parser.add_argument("-s","--show", action='store_true', dest="SHOW",
                            default=False, help="show results in default web browser [NO]")
        
        parser.add_argument('-o', '--output', required=False, dest="OUTPUT",
                            default='Plots', metavar='Plot directory', type=str,
                            help="Name of the directory where plots are generated.")
        
        parser.add_argument('infile', nargs='+', type=FileType('r'),
                            default=sys.stdin, metavar='histo',
                            help='A POWHEG-BOX histogram file.')
        
        #self.action = []
        args = parser.parse_args()
        self.overwrite = args.OVERWRITE
        self.show      = args.SHOW
        self.outdir    = args.OUTPUT
        self.files     = args.infile


if __name__ == '__main__':

    # list of histograms
    histos = []

    # read input parameters
    param = InParams()

    # fill histogram list
    for f in param.files:
        histos.append(RawHist.openfile(f.name))

    histoplot = HistPlot(histos)
    histoplot.overwrite = param.overwrite
    histoplot.outdir = param.outdir
    histoplot.html = True
    histoplot.show = param.show
    histoplot.plot()

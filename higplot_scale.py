#! /usr/bin/env python

import os
import sys
import re
import gzip
#import glob

from highisttools import *
from argparse import ArgumentParser, FileType, ArgumentTypeError

def trihisto(s):
    try:
        h_c, h_u, h_d = s.split(',')
        return [h_c, h_u, h_d]
    except:
        raise ArgumentTypeError("Histograms must be passed in triplets as central_scale, scale_up, scale_down")


class InParams:
    DEFAULT_ACTION = 'print_xs'

        
    def __init__(self):
        
        parser = ArgumentParser(description='Plots histogram "histo"')
    
        parser.add_argument('-w', '--overwrite', action='store_true', dest="OVERWRITE",
                            default=False, help="whether to overwrite an existing folder [NO].")
        
        parser.add_argument("-s","--show", action='store_true', dest="SHOW",
                            default=False, help="show results in default web browser [NO]")
        
        parser.add_argument('-o', '--output', required=False, dest="OUTPUT",
                            default='Plots', metavar='Plot directory', type=str,
                            help="Name of the directory where plots are generated.")
                
        parser.add_argument('infile', nargs='+', type=trihisto, metavar='histo',
                            help='A triplet of POWHEG-BOX histogram files.')

        
        #self.action = []
        args = parser.parse_args()
        self.overwrite = args.OVERWRITE
        self.show      = args.SHOW
        self.outdir    = args.OUTPUT
        self.files     = args.infile


if __name__ == '__main__':

    # list of histograms
    triplets = []

    # read input parameters
    param = InParams()
    
    # fill histogram list
    print("Found %i histogram triplets:" % len(param.files))

    for t in param.files:
        print t
        triplets.append(TriHist(t))

    histoplot = HistPlotScale(triplets)
    histoplot.overwrite = param.overwrite
    histoplot.outdir = param.outdir
    histoplot.html = True
    histoplot.show = param.show
    histoplot.plot()

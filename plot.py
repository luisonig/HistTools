#! /usr/bin/env python

import sys
from pwghisttool import *



if __name__ == '__main__':
    ## Argument parser
    from argparse import ArgumentParser, FileType

    parser = ArgumentParser(description='Prints histogram "histo"')
    
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

    args = parser.parse_args()
    file = args.infile

    # list of histograms:
    histos=[]
    
    for f in file:
        histos.append(PwgHist(f))

    myplot=PwgPlot(histos)
    myplot.outdir = args.OUTPUT
    myplot.overwrite = args.OVERWRITE
    myplot.html = True
    myplot.show = args.SHOW
    myplot.plot()

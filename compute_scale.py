#! /usr/bin/env python

import sys, os
from pwghisttool import *

def scalevariation(files, scale, suffix, nloonly):
    
    dir_lo  = os.path.split(files[0].name)[0]
    dir_nlo = os.path.split(files[1].name)[0]

    file_lo  = os.path.splitext(os.path.split(files[0].name)[1])[0]
    file_nlo = os.path.splitext(os.path.split(files[1].name)[1])[0]
    
    hist_lo  = PwgHist(files[0])
    hist_nlo = PwgHist(files[1])

    hist_lo_up  = hist_lo.scalevariation_lo(scale,2.0*scale,3)
    hist_lo_dn  = hist_lo.scalevariation_lo(scale,0.5*scale,3)
    hist_nlo_up = hist_nlo.scalevariation_nlo(hist_lo,scale,2.0*scale,3)
    hist_nlo_dn = hist_nlo.scalevariation_nlo(hist_lo,scale,0.5*scale,3)


    if not nloonly:
        file=open(dir_lo+'/'+file_lo.replace('r1','r2')+'-'+suffix+'.top','w')
        hist_lo_up.write(file)
        file=open(dir_lo+'/'+file_lo.replace('r1','r.5')+'-'+suffix+'.top','w')
        hist_lo_dn.write(file)

    file=open(dir_nlo+'/'+file_nlo.replace('r1','r2')+'-'+suffix+'.top','w')
    hist_nlo_up.write(file)
    file=open(dir_nlo+'/'+file_nlo.replace('r1','r.5')+'-'+suffix+'.top','w')
    hist_nlo_dn.write(file)



if __name__ == '__main__':

    ## Argument parser
    from argparse import ArgumentParser, FileType

    parser = ArgumentParser(description='Given a central fixed scale, a LO and a NLO histogram,\
    computes the scale varied histograms by factors of 0.5 and 2, and writes them to files.')

    parser.add_argument('-n', '--nloonly', action='store_true', dest="NLOONLY",
                        default=False, help="whether to perform scale variation just for NLO [NO].")
    
    parser.add_argument('-c', '--central', nargs=1, required=True, dest="CENTRAL",
                        default=100, metavar='central-scale', type=float,
                        help="central scale value [100 (GeV)]")

    parser.add_argument('infiles', nargs=2, type=FileType('r'),
                        default=sys.stdin, metavar='histo',
                        help='A LO and a NLO histogram file (order matters!).')

    args  = parser.parse_args()
    files = args.infiles
    scale = args.CENTRAL[0]
    nloonly = args.NLOONLY

    
    suffix = 'computed'

    scalevariation(files, scale, suffix, nloonly)

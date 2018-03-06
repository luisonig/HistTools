#!/usr/bin/env python

import sys, os
import subprocess
import copy
import re
import numpy as np
import logging
import cStringIO
import fnmatch
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Iterable
from jinja2 import Environment, FileSystemLoader, Template

__all__ = ['RawHist','TriHist', 'HistPlot', 'HistPlotScale']

class RawHist:

    def __init__(self, f=None, name=None, yodaotic=False):
        self.horder = []
        self.hdata = {}
        self.hist2d = {}
        self.count = 0
        self.binned = 0
        self.name = ''
        if f:
            self.name = f.name
            if yodaotic:
                self.name=name
            self.fromfile(f, yodaotic)
    
    @staticmethod
    def latex(hn, head, *hlist):
        s = r"""
\begin{table}\centering
  \begin{tabular}{ccc}
    %% %s
    \hline \\
    %s
    \hline
""" % (hn, head)
        totalxs = [0.] * len(hlist)
        for i in range(len(hlist[0].hdata[hn][0])):
            (x1,x2,y,yp,ym) = hlist[0].hdata[hn][...,i]
            if x2 < 0:
                s += "   $%.1f$ --- $(%.1f)$" % (x1, x2)
            else:
                s += "   $%.1f$ --- $%.1f$" % (x1, x2)
            width = x2 - x1
            for hi in range(len(hlist)):
                (nx1,nx2,y,yp,ym) = hlist[hi].hdata[hn][...,i]
                assert x1 == nx1 and x2 == nx2
                val, err = y*width, 0.5*(yp+ym)*width
                totalxs[hi] += val
                if 0.1 < err:
                    s += " & $%.1f$ $(%.1f)$" % (val, err)
                elif 0.01 < err:
                    s += " & $%.2f$ $(%.2f)$" % (val, err)
                elif 0.001 < err:
                    s += " & $%.3f$ $(%.3f)$" % (val, err)
                elif 0.0001 < err:
                    s += " & $%.4f$ $(%.4f)$" % (val, err)
                elif 0.00001 < err:
                    s += " & $%.5f$ $(%.5f)$" % (val, err)
            s += " \\\\\n"
        s += r"""\hline
  %% total XS = %s
  \end{tabular}
\end{table}
""" % ', '.join("%f" % xs for xs in totalxs)
        return s

    def align(self, names, edges, merge=1, relative=False):
        for hn,(low,high) in zip(names, edges):
            hdhn = self.hdata[hn]
            if low not in hdhn[0]:
                if low < hdhn[0][0]:
                    raise RuntimeError("stretching left unimplemented")
                fi = len([x for x in hdhn[0] if x < low])-1
                if fi != 0:
                    continue
                ti = fi + merge
                newwidth = hdhn[1][ti]-low
                if relative:
                    newwidth = hdhn[1][ti]-hdhn[0][fi]
                area = sum((hdhn[1][i]-hdhn[0][i])*hdhn[2][i] for i in range(fi,ti+1))
                errarea1 = sum((hdhn[1][i]-hdhn[0][i])*hdhn[3][i] for i in range(fi,ti+1))
                errarea2 = sum((hdhn[1][i]-hdhn[0][i])*hdhn[4][i] for i in range(fi,ti+1))
                newheight = area/newwidth
                hdhn[0][ti] = low
                hdhn[2][ti] = area/newwidth
                hdhn[3][ti] = errarea1/newwidth
                hdhn[4][ti] = errarea2/newwidth
                self.hdata[hn] = np.delete(hdhn, range(0, ti), 1)
        return None

    @staticmethod
    def dictalign(histdict, names, edges, merge=1, relative=False):
        newdict = {}
        for k,hist in histdict.items():
            assert isinstance(hist, RawHist)
            newdict[k] = hist.clone()
            newdict[k].align(names, edges, merge, relative)
        return newdict

    @staticmethod
    def listalign(histlist, names, edges, merge=1, relative=False):
        newlist = []
        for k in range(len(histlist)):
            assert isinstance(histlist[k], RawHist)
            newlist.append(histlist[k].clone())
            newlist[k].align(names, edges, merge, relative)
        return newlist

    @staticmethod
    def averagedext(histlist):
        new = histlist[0].clone()
        for hn in histlist[0].hdata.keys():
            w = sum(histlist[i].hdata[hn][2] for i in range(len(histlist)))
            w2 = sum(np.square(histlist[i].hdata[hn][2]) for i in range(len(histlist)))
            n = len(histlist)
            new.hdata[hn][2] = w/n
            err = np.sqrt((w2 - w*w/n)/(n-1))
            new.hdata[hn][3] = err
            new.hdata[hn][4] = err
        return new

    @staticmethod
    def averagedperfectly(histlist):
        new = histlist[0].clone()
        new.count = sum(h.count for h in histlist)
        new.binned = sum(h.binned for h in histlist)
        hists = [h for h in histlist if h.count > 0]  # filter empty histograms
        if len(histlist) != len(hists):
            print "Warning: 'averagedperfectly' ignored %d histograms with zero count" % (len(histlist) - len(hists))
        for hn in histlist[0].hdata.keys():
            new.hdata[hn][2] = sum(h.hdata[hn][2]*h.count for h in hists)/new.count
            new.hdata[hn][3] = np.sqrt(sum(np.square(h.hdata[hn][3]*h.count) for h in hists))/new.count
            new.hdata[hn][4] = np.sqrt(sum(np.square(h.hdata[hn][4]*h.count) for h in hists))/new.count
        return new

    @staticmethod
    def summed(histlist):
        new = histlist[0].clone()
        for hn in histlist[0].hdata.keys():
            new.hdata[hn][2] = sum(histlist[i].hdata[hn][2] for i in range(len(histlist)))
            new.hdata[hn][3] = np.sqrt(sum(np.square(histlist[i].hdata[hn][3]) for i in range(len(histlist))))
            new.hdata[hn][4] = np.sqrt(sum(np.square(histlist[i].hdata[hn][4]) for i in range(len(histlist))))
        return new
    
    @staticmethod
    def sqsummed(histlist):
        new = histlist[0].clone()
        for hn in histlist[0].hdata.keys():
            new.hdata[hn][2] = sum(histlist[i].hdata[hn][2]**2 for i in range(len(histlist)))
            new.hdata[hn][3] = np.array([0]*len(new.hdata[hn][0]))
            new.hdata[hn][4] = np.array([0]*len(new.hdata[hn][0]))
        return new
    
    @staticmethod
    def averaged(histlist):
        return RawHist.summed(histlist)/len(histlist)

    @staticmethod
    def openfile(fname):
        if fname.endswith('.aida'):
            pipe = subprocess.Popen(["aida2flat", fname], stdout=subprocess.PIPE).stdout
            return RawHist(pipe, fname, yodaotic=True)
        if fname.endswith('.yoda'):
            pipe = subprocess.Popen(["yoda2flat", fname, "-"], stdout=subprocess.PIPE).stdout
            return RawHist(pipe, fname, yodaotic=True)
        else:
            with open(fname, 'r') as f:
                return RawHist(f)

    @staticmethod
    def extractfile(tar, fname):
        f = tar.extractfile(fname)
        rh = RawHist(f)
        f.close()
        return rh

    def clone(self):
        new = copy.deepcopy(self)
        new.binned = 0
        new.count = 0
        return new

    def fromfile(self, f, yodaotic):
        histname2d = None
        contents2d = None
        edges2d = None
        state = 0
        for line in f:
            if state == 0 and line[:20] == '# BEGIN HISTOGRAM2D ':
                m = re.match(r"^/([^/]+)/(.*?)\s*$", line[20:])
                if m:
                    analysis, histname2d = m.group(1), m.group(2)
                    contents2d = []
                else:
                    print line
                    raise RuntimeError("Can't not parse 2D histogram name")
                state = 10
            elif state == 10:
                sio = cStringIO.StringIO()
                sio.write(line[1:])
                sio.seek(0)
                edges2d = np.loadtxt(sio, unpack=True)
                sio = None
                state = 0
            elif state == 0 and line[:17] == '# END HISTOGRAM2D':
                self.hist2d[histname2d] = (edges2d, contents2d)
                histname2d = None
                contents2d = None
                edges2d = None
            elif state == 0 and line[:18] == '# BEGIN HISTOGRAM ':
                m = re.match(r"^/([^/]+)/(.*?)\s*$", line[18:])
                if m:
                    analysis, histname = m.group(1), m.group(2)
                    if yodaotic:
                        histname = analysis + '/' + histname
                    if histname2d is not None:
                        assert histname.startswith(histname2d)
                        contents2d.append(histname)
                else:
                    print line
                    raise RuntimeError("Can't not parse histogram name")
                sio = cStringIO.StringIO()
                nbins = 0
                state = 1
            elif state == 0 and line[:16] == '# BEGIN HISTO1D ':
                m = re.match(r"^/([^/]+)/(.*?)\s*$", line[16:])
                if m:
                    analysis, histname = m.group(1), m.group(2)
                    if yodaotic:
                        histname = analysis + '_' + histname
                    if histname2d is not None:
                        assert histname.startswith(histname2d)
                        contents2d.append(histname)
                else:
                    print line
                    raise RuntimeError("Can't not parse histogram name")
                sio = cStringIO.StringIO()
                nbins = 0
                state = 1
            elif state == 1:
                if line[0] != '#':
                    if yodaotic:
                        m = re.match(r"^Title=(.*?)\s*$", line)
                        if m:
                            subname = m.group(1)
                            if subname:
                                assert histname2d is None
                                histname = histname + ':' + subname
                else:
                    state = 2
            elif state == 2:
                if line[0] == '#':
                    m = re.match(r"^## Num bins: (\d+)\s*$", line)
                    if m:
                        nbins = int(m.group(1))
                else:
                    sio.write(line)
                    state = 3
            elif state == 3:
                if line[0] != '#':
                    sio.write(line)
                else:
                    sio.seek(0)
                    while histname in self.hdata:
                        histname += ':dup'
                    self.horder.append(histname)
                    self.hdata[histname] = np.loadtxt(sio, unpack=True)
                    if len(self.hdata[histname].shape) == 1:
                        self.hdata[histname] = np.array([[x] for x in self.hdata[histname]])
                    assert yodaotic or len(self.hdata[histname][0]) == nbins
                    sio = None
                    state = 0
            elif state == 0 and line[:14] == '### Finalize: ':
                m = re.match(r"^### Finalize: groups (\d+), events (\d+)"
                             r" \(binned (\d+)\) \[trials (\d+)\]$", line)
                if m:
                    groups, events, binned, trials = map(float, m.groups())
                    self.count = trials
                    self.binned = binned

        f.close()

    def tofile(self, fname):
        with open(fname, 'w') as f:
            f.write(str(self))

    def __str__(self, sort=False):
        sio = cStringIO.StringIO()
        if sort:
            horder = sorted(self.horder)
        else:
            horder = self.horder
        for histname in horder:
            data = self.hdata[histname]
            sio.write("# BEGIN HISTOGRAM /Analysis/%s\n" % histname)
            area = sum((data[1]-data[0])*data[2])
            sio.write("## Area: %.15e\n" % area)
            sio.write("## Num bins: %d\n" % len(data[0]))
            sio.write("## xlow   xhigh     val     errminus  errplus\n")
            np.savetxt(sio, np.transpose(data))
            sio.write("# END HISTOGRAM\n\n\n")
        return sio.getvalue()

    def __add__(self, other):
        new = self.clone()
        for hn in self.hdata.keys():
            assert (self.hdata[hn][:2] == other.hdata[hn][:2]).all()
            (y0, eu0, ed0) = self.hdata[hn][2:]
            (y1, eu1, ed1) = other.hdata[hn][2:]
            new.hdata[hn][2] = y0 + y1
            new.hdata[hn][3] = np.sqrt(eu0*eu0 + eu1*eu1)
            new.hdata[hn][4] = np.sqrt(ed0*ed0 + ed1*ed1)
        return new

    def __sub__(self, other):
        new = self.clone()
        for hn in self.hdata.keys():
            assert (self.hdata[hn][:2] == other.hdata[hn][:2]).all()
            (y0, eu0, ed0) = self.hdata[hn][2:]
            (y1, eu1, ed1) = other.hdata[hn][2:]
            new.hdata[hn][2] = y0 - y1
            new.hdata[hn][3] = np.sqrt(eu0*eu0 + eu1*eu1)
            new.hdata[hn][4] = np.sqrt(ed0*ed0 + ed1*ed1)
        return new

    def __mul__(self, other):
        new = self.clone()
        for hn in self.hdata.keys():
            new.hdata[hn][2] = self.hdata[hn][2] * other
            new.hdata[hn][3] = self.hdata[hn][3] * other
            new.hdata[hn][4] = self.hdata[hn][4] * other
        return new

    @staticmethod
    def maximum(histlist):
        new = histlist[0].clone()
        for hn in histlist[0].hdata.keys():
            for i in range(len(histlist)):
                maxvec = np.argmax(np.stack((histlist[i].hdata[hn][2], new.hdata[hn][2])), axis=0)
                new.hdata[hn][2] = maxvec*new.hdata[hn][2] + (1.0-maxvec)*histlist[i].hdata[hn][2]
                new.hdata[hn][3] = maxvec*new.hdata[hn][3] + (1.0-maxvec)*histlist[i].hdata[hn][3]
                new.hdata[hn][4] = maxvec*new.hdata[hn][4] + (1.0-maxvec)*histlist[i].hdata[hn][4]
        return new

    @staticmethod
    def minimum(histlist):
        new = histlist[0].clone()
        for hn in histlist[0].hdata.keys():
            for i in range(len(histlist)):
                minvec = np.argmin(np.stack((histlist[i].hdata[hn][2], new.hdata[hn][2])), axis=0)
                new.hdata[hn][2] = minvec*new.hdata[hn][2] + (1.0-minvec)*histlist[i].hdata[hn][2]
                new.hdata[hn][3] = minvec*new.hdata[hn][3] + (1.0-minvec)*histlist[i].hdata[hn][3]
                new.hdata[hn][4] = minvec*new.hdata[hn][4] + (1.0-minvec)*histlist[i].hdata[hn][4]
        return new

    
    def square(self):
        new = self.clone()
        for hn in self.hdata.keys():
            new.hdata[hn][2] = np.square(self.hdata[hn][2])
            new.hdata[hn][3] = np.square(self.hdata[hn][3])
            new.hdata[hn][4] = np.square(self.hdata[hn][4])
        return new

    def sqrt(self):
        new = self.clone()
        for hn in self.hdata.keys():
            new.hdata[hn][2] = np.sqrt(self.hdata[hn][2])
            new.hdata[hn][3] = np.sqrt(self.hdata[hn][3])
            new.hdata[hn][4] = np.sqrt(self.hdata[hn][4])
        return new

    def __div__(self, other):
        new = self.clone()
        if isinstance(other, RawHist):
            for hn in self.hdata.keys():
                assert (self.hdata[hn][:2] == other.hdata[hn][:2]).all()
                (y0, eu0, ed0) = self.hdata[hn][2:]
                (y1, eu1, ed1) = other.hdata[hn][2:]
                new.hdata[hn][2] = y0/y1
                new.hdata[hn][3] = np.sqrt(eu0*eu0*y1*y1 + eu1*eu1*y0*y0)/(y1*y1)
                new.hdata[hn][4] = np.sqrt(ed0*ed0*y1*y1 + ed1*ed1*y0*y0)/(y1*y1)
        else:
            for hn in self.hdata.keys():
                new.hdata[hn][2] = self.hdata[hn][2] / other
                new.hdata[hn][3] = self.hdata[hn][3] / other
                new.hdata[hn][4] = self.hdata[hn][4] / other
        return new

    def __eq__(self, other):
        if not isinstance(other, RawHist):
            return NotImplemented
        for hn in self.hdata.keys():
            if not (self.hdata[hn] == other.hdata[hn]).all():
                return False
        return True

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def merge(self, hn, pairs):
        for (a,b) in pairs:
            val = 0.
            err1 = 0.
            err2 = 0.
            for i in range(a, a+b):
                width = self.hdata[hn][1][i]-self.hdata[hn][0][i]
                val += width*self.hdata[hn][2][i]
                tmp = width*self.hdata[hn][3][i]
                err1 += tmp*tmp
                tmp = width*self.hdata[hn][4][i]
                err2 += tmp*tmp
            err1 = sqrt(err1)/b
            err2 = sqrt(err2)/b

            self.hdata[hn][1][a] = self.hdata[hn][1][a+b-1]
            width = self.hdata[hn][1][a]-self.hdata[hn][0][a]
            self.hdata[hn][2][a] = val/width
            self.hdata[hn][3][a] = err1/width
            self.hdata[hn][4][a] = err2/width
            self.hdata[hn] = np.delete(self.hdata[hn], range(a+1, a+b), 1)

# -----------------------------------------------------------------------------

class TriHist:
    def __init__(self, histos=None):
        self.unsorted = []
        self.horder = []
        self.hdata = {}
        self.name = ''
        if histos:
            assert isinstance(histos, list)
            assert len(histos)==3
            for f in histos:
                assert isinstance(f,str)
                self.unsorted.append(RawHist.openfile(f))
            self.name = histos[0]
            self.combine()

                
    def combine(self):
        maximum = RawHist.maximum(self.unsorted)
        minimum = RawHist.minimum(self.unsorted)
        #print "-------"
        #print self.unsorted[0].hdata
        #print "-------"
        for hn in self.unsorted[0].horder:
            self.horder.append(hn)
            self.hdata[hn] = np.append(self.unsorted[0].hdata[hn],maximum.hdata[hn][2:],axis=0)
            self.hdata[hn] = np.append(self.hdata[hn],minimum.hdata[hn][2:],axis=0)
        #print self.hdata
        return
            
# -----------------------------------------------------------------------------

class HistPlot:
    def __init__(self, histo=None):
        loglevel=logging.INFO
        logging.basicConfig(level=loglevel, format="%(message)s")

        self.hlist = []
        self.outdir = 'plots'
        self.format = 'png'
        self.html = False
        self.overwrite = False
        self.show = False
        self.plotdata = []
        
        if histo:
            if isinstance(histo,list):
                for i in list(histo):
                    assert isinstance(i, RawHist)
                    self.hlist.append(i)
            else:
                self.hlist.append(histo)

    def _createdir(self, outdir):
        if os.path.exists(outdir) and os.path.isdir(outdir):
            if self.overwrite==False:
                logging.error("Output directory "+outdir+" already existing.\n")
                logging.error("Call again with options -w or --overwrite to overwrite it.\n")
                sys.exit(2)
            else:
                import shutil
                shutil.rmtree(outdir)
                os.makedirs(outdir)
        else:
            if os.path.exists(outdir):
                logging.error(outdir+" exists but is not a valid directory.\n")
                logging.error("Please check your inputs.\n")
                sys.exit(2)
            else:
                os.makedirs(outdir)
        os.system("chmod 755 "+ outdir)

    def plot(self):
        if len(self.hlist) < 8:
            color_cycle = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        else:
            colormap = plt.cm.rainbow
            color_cycle = [colormap(i) for i in np.linspace(0, 0.9, len(self.hlist))]

        self._createdir(self.outdir)

        font = {'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 18}
        plt.rc('font', **font)
        
        # Now make plots:
        for title in self.hlist[0].horder:
            figure=plt.figure(figsize = (10,10), dpi=100)
            ax1=plt.subplot2grid((5,4),(0,0),colspan=4,rowspan=4)
            ax2=plt.subplot2grid((5,4),(4,0),colspan=4,rowspan=1,sharex=ax1)
            ax1.grid(True,which='both', axis='both',alpha=0.2)
            ax2.grid(True,which='both', axis='both',alpha=0.2)
            y_min=sys.float_info.max
            y_max=-y_min
            numFound=0
            for i in range(0,len(self.hlist)):
                if title in self.hlist[i].hdata:
                    numFound+=1
                    print("Found plot "+title.rstrip("\n")+" in "+self.hlist[i].name.rstrip("\n"))
                    data=self.hlist[i].hdata[title]
                    self._AddToPlotdata(self.hlist[i].name.rstrip("\n"), title.rstrip("\n"), data)
                    (xl,xh,y,erry,erryd)=data
                    if numFound == 1:
                        valueRef=y
                        errorRef=erry
                    legendEntry=self.hlist[i].name.rstrip("\n")
                    line=ax1.plot(np.transpose([xl,xh]).flatten(),np.transpose([y,y]).flatten(),linewidth=0.7,label=legendEntry, c=color_cycle[i])
                    ax1.errorbar(0.5*(xl+xh), y, yerr = erry , fmt="none", capsize=1, elinewidth=0.3, ecolor=color_cycle[i])

                    low_envelope =  np.minimum.reduce(y)
                    high_envelope = np.maximum.reduce(y)

                    if high_envelope > 0:
                        y_max = max(y_max,np.power(10,np.ceil(np.log10(high_envelope))))
                    else:
                        y_max = max(y_max,1.1*np.ceil(np.array(high_envelope).max()))

                    if low_envelope > 0:
                        y_min = min(y_min,np.power(10,np.floor(np.log10(low_envelope))))                      
                    elif low_envelope == 0:
                        y_min = 0.0
                    else:
                        y_min = min(y_min,0.8*np.floor(np.array(low_envelope).min()))
                        
                    if isinstance(y, Iterable):
                        value_ratio=[a/b if b!=0.0 else 0.0  for (a,b) in zip(y,valueRef)]
                        error_ratio=[a/b*np.sqrt(np.power(da/a,2)+np.power(db/b,2)) if (a!=0.0 and b!=0.0) else 0.0  for (a,da,b,db) in zip(y,erry,valueRef,errorRef)]
                    else:
                        value_ratio = (y/valueRef) if valueRef!=0.0 else 0.0
                        error_ratio = (erry/valueRef) if valueRef!=0.0 else 0.0

                    ax2.plot(np.transpose([xl,xh]).flatten(),np.transpose([value_ratio,value_ratio]).flatten(),lw=0.7,c=color_cycle[i])
                    ax2.errorbar(0.5*(xl+xh), value_ratio, yerr=error_ratio, fmt="none", capsize=1, ecolor=color_cycle[i], elinewidth=0.3)
                    if numFound > 0:
                        leg=ax1.legend(loc='best',fancybox=True)
                        leg.get_frame().set_alpha(0.5)
                    if (y_min >= 0 and y_max > 0) :
                        ax1.set_yscale("log", nonposy='clip',subsy=[1,2,3,4,5,6,7,8,9])
                        ax1.yaxis.set_major_locator(mticker.LogLocator())
                        ax1.yaxis.set_minor_locator(mticker.LogLocator(subs=[1.0,2.0,3,0,4.0,5.0,6.0,7.0,8.0,9.0]))
                        ax1.set_ylim(y_min,y_max)
                    #ax1.set_ylabel(r'$d \sigma /d x \ [\rm{pb/unit}]$')
                    ax1.tick_params(axis='x',labelbottom='off')
                    #ax2.set_ylabel(r'frac. diff.')
                    ax2.set_ylabel(r'ratio')
                    ax2.set_xlabel(title, fontsize = 30)
                    plt.subplots_adjust(hspace=0.0)
                    #plt.tight_layout(h_pad=0.0)
                    outfile=self.outdir+"/"+title.replace(" ","_")
                    self._saveFigure(figure,outfile,self.format)
            self._generatePlotfile(self.outdir,title, 'plottemplate.py')

        if self.html:
            self._WriteHTML(self.outdir,self.hlist[0].horder,self.format)

        if self.show:
            import webbrowser
            webbrowser.open(self.outdir+"/index.html")

        
            
    def _saveFigure(self, fig, path, ext='png', close=True):
        """Save a figure from pyplot.

        Parameters
        ----------
        fig : matplotlib.figure
        The figure object to be saved

        path : string
        The path (and filename, without the extension) to save the
        figure to.

        ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

        close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.

        """

        # Extract the directory and filename from the given path
        directory = os.path.split(path)[0]
        filename = "%s.%s" % (os.path.split(path)[1], ext)
        if directory == '':
            directory = '.'

        # If the directory does not exist, create it
        if (not os.path.exists(directory)) and (not os.path.isdir(directory)):
            logging.error("Directory %s does not exist"%directory)

        # The final path to save to
        savepath = os.path.join(directory, filename)

        logging.debug("Saving figure to '%s'..." % savepath)

        # Actually save the figure
        fig.savefig(savepath,dpi=fig.dpi)

        # Close it
        if close:
            plt.close()

        logging.debug("Done with '%s'"%path)

    def _WriteHTML(self, outdir, obslist, ext='png'):
        '''
        Write the output HTML file for a given set of plots
        '''
        # open file
        HTMLfile = open(outdir+'/index.html','w')
        # headers
        HTMLfile.write('<html>'+'\n')
        # title
        HTMLfile.write('<head>')
        HTMLfile.write('<title> '+outdir+' </title>'+'\n')
        HTMLfile.write('<style> \n  html { font-family: sans-serif; } \n  img { border: 0; } \n  a { text-decoration: none; font-weight: bold; } \n </style></head>'+'\n')
        HTMLfile.write('<body bgcolor=white>'+'\n')
        HTMLfile.write('<h1> '+outdir+'</h1>')
        # date
        import datetime
        from dateutil.tz import tzlocal
        now = datetime.datetime.now(tzlocal())
        fmt = now.strftime('%Y-%m-%d at %H:%M:%S %Z')
        HTMLfile.write('<em>This page was created on ' + fmt  + '.</em>\n')

        HTMLfile.write('<h3> Plots  </h3>'+'\n')
        HTMLfile.write('<div>'+'\n')
        # look in outdir and find all png files
        files = os.listdir(outdir)
        plots = []
        for file in files:
            if fnmatch.fnmatch(file,"*"+ext):
                plots.append(file)
        # plots
        for obs in obslist:
            if not any(obs.replace(" ","_") in plt for plt in plots):
                sys.stderr.write("Could not find plot for "+obs+'\n')
            else:
                HTMLfile.write(self._ImageTag(obs.replace(" ","_")+"."+ext))
        HTMLfile.write('</div>'+'\n')
        # footers
        HTMLfile.write('</body>'+'\n')
        HTMLfile.write('</html>'+'\n')
        # close file
        HTMLfile.close()

    def _ImageTag(self, plot):
        '''
        Returns a HTML command to include an image on a webpage.
        '''
        title, ext = os.path.splitext(plot)
        string  = '<div style=\"float:left; font-size:smaller; font-weight:bold;\">'+'\n'
        string += '<a href=\"'+title+'-source.py\">&#8984</a>'+'\n'
        string += '<a href=\"#'+plot+'\">&#9875;</a> '+plot.rpartition('.')[0]+':<br>'+'\n'
        string += '<a name=\"'+plot+'\"><a href=\"'+plot+'\">'+'\n'
        string += '<img HEIGHT=500 src=\"'+plot+'\">'+'\n'+'</a></a>'+'\n'+'</div>'+'\n'
        return string


    def _AddToPlotdata(self, filename, title, data):
        template = Template("    plot.addData('{{ filename }}',\n                 {'{{ title }}': np.transpose(\n\t\t     {{ data }} )})")
        np.set_printoptions(edgeitems=5,linewidth=150,threshold=np.nan)
        self.plotdata.append(template.render(filename=filename,
                                             title=title,
                                             data=np.array2string(np.transpose(data), separator=',', prefix='                    ')))
    
    def _generatePlotfile(self, outdir, title, template):
        path=os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(path))
        plotfile = open(outdir+'/'+title+'-source.py','w')
        template = env.get_template(template)
        data = ''
        #print self.plotdata
        for dataset in self.plotdata:
            data += dataset+"\n\n"
        plotfile.write(template.render(title=title,data=data))
        del self.plotdata[:]

# -----------------------------------------------------------------------------        

class HistPlotScale(HistPlot):
    def __init__(self, histotriplets=None):
        loglevel=logging.INFO
        logging.basicConfig(level=loglevel, format="%(message)s")

        self.hlist = []
        self.outdir = 'plots'
        self.format = 'png'
        self.html = False
        self.overwrite = False
        self.show = False
        self.plotdata = []
        
        if histotriplets:
            if isinstance(histotriplets,list):
                for i in histotriplets:
                    assert isinstance(i, TriHist)
                    self.hlist.append(i)
            else:
                assert isinstance(histotriplets, TriHist)
                self.hlist.append(histotriplets)

    def plot(self):
        if len(self.hlist) < 8:
            color_cycle = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        else:
            colormap = plt.cm.rainbow
            color_cycle = [colormap(i) for i in np.linspace(0, 0.9, len(self.hlist))]
            
        self._createdir(self.outdir)

        font = {'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 18}
        plt.rc('font', **font)

        # Now make plots:
        for title in self.hlist[0].horder:
            figure=plt.figure(figsize = (10,10), dpi=100)
            ax1=plt.subplot2grid((5,4),(0,0),colspan=4,rowspan=4)
            ax2=plt.subplot2grid((5,4),(4,0),colspan=4,rowspan=1,sharex=ax1)
            ax1.grid(True,which='both', axis='both',alpha=0.2)
            ax2.grid(True,which='both', axis='both',alpha=0.2)
            y_min=sys.float_info.max
            y_max=-y_min
            numFound=0
            for i in range(0,len(self.hlist)):
                if title in self.hlist[i].hdata:
                    numFound+=1
                    print("Found plot "+title.rstrip("\n")+" in "+self.hlist[i].name.rstrip("\n"))
                    data=self.hlist[i].hdata[title]
                    self._AddToPlotdata(self.hlist[i].name.rstrip("\n"), title.rstrip("\n"), data)
                    (xl,xh,y,erry,errya,yup,erryu,erryua,ydown,errydown,errydowna)=data
                    if numFound == 1:
                        valueRef=y
                        errorRef=erry
                    legendEntry=self.hlist[i].name.rstrip("\n")
                    xval = np.transpose([xl,xh]).flatten()
                    ax1.plot(xval, np.transpose([y,y]).flatten(),linewidth=0.7,label=legendEntry, c=color_cycle[i])
                    ax1.errorbar(0.5*(xl+xh), y, yerr = erry , fmt="none", capsize=1, elinewidth=0.3, ecolor=color_cycle[i])
                    ax1.fill_between(xval, np.transpose([ydown, ydown]).flatten(), np.transpose([yup, yup]).flatten(), color=color_cycle[i], alpha=0.2, lw=0.7)
                    ax1.plot(xval, np.transpose([ydown, ydown]).flatten(), color=color_cycle[i], lw=0.7)
                    ax1.plot(xval, np.transpose([yup, yup]).flatten(), color=color_cycle[i], lw=0.7)
                    
                    low_envelope =  np.minimum.reduce(ydown)
                    high_envelope = np.maximum.reduce(yup)

                    if high_envelope > 0:
                        y_max = max(y_max,np.power(10,np.ceil(np.log10(high_envelope))))
                    else:
                        y_max = max(y_max,1.1*np.ceil(np.array(high_envelope).max()))

                    if low_envelope > 0:
                        y_min = min(y_min,np.power(10,np.floor(np.log10(low_envelope))))                      
                    elif low_envelope == 0:
                        y_min = 0.0
                    else:
                        y_min = min(y_min,0.8*np.floor(np.array(low_envelope).min()))
                        
                    if isinstance(y, Iterable):
                        value_ratio = [a/b if b!=0.0 else 0.0  for (a,b) in zip(y,valueRef)]
                        error_ratio = [a/b*np.sqrt(np.power(da/a,2)+np.power(db/b,2)) if (a!=0.0 and b!=0.0) else 0.0  for (a,da,b,db) in zip(y,erry,valueRef,errorRef)]
                        sc_up_ratio = [a/b if b!=0.0 else 0.0  for (a,b) in zip(yup,valueRef)]
                        sc_down_ratio = [a/b if b!=0.0 else 0.0  for (a,b) in zip(ydown,valueRef)]
                    else:
                        value_ratio = (y/valueRef) if valueRef!=0.0 else 0.0
                        error_ratio = (erry/valueRef) if valueRef!=0.0 else 0.0
                        sc_up_ratio = (yup/valueRef) if valueRef!=0.0 else 0.0
                        sc_down_ratio = (ydown/valueRef) if valueRef!=0.0 else 0.0

                    ax2.plot(xval,np.transpose([value_ratio,value_ratio]).flatten(),lw=0.7,c=color_cycle[i])
                    ax2.errorbar(0.5*(xl+xh), value_ratio, yerr=error_ratio, fmt="none", capsize=1, ecolor=color_cycle[i], elinewidth=0.3)
                    ax2.fill_between(xval, np.transpose([sc_down_ratio, sc_down_ratio]).flatten(), np.transpose([sc_up_ratio, sc_up_ratio]).flatten(), color=color_cycle[i], alpha=0.2, lw=0.7)
                    ax2.plot(xval, np.transpose([sc_down_ratio, sc_down_ratio]).flatten(), color=color_cycle[i], lw=0.7)
                    ax2.plot(xval, np.transpose([sc_up_ratio, sc_up_ratio]).flatten(), color=color_cycle[i], lw=0.7)
                    if numFound > 0:
                        leg=ax1.legend(loc='best',fancybox=True)
                        leg.get_frame().set_alpha(0.5)
                    if (y_min >= 0 and y_max > 0) :
                        ax1.set_yscale("log", nonposy='clip',subsy=[1,2,3,4,5,6,7,8,9])
                        ax1.yaxis.set_major_locator(mticker.LogLocator())
                        ax1.yaxis.set_minor_locator(mticker.LogLocator(subs=[1.0,2.0,3,0,4.0,5.0,6.0,7.0,8.0,9.0]))
                        ax1.set_ylim(y_min,y_max)
                    #ax1.set_ylabel(r'$d \sigma /d x \ [\rm{pb/unit}]$')
                    ax1.tick_params(axis='x',labelbottom='off')
                    #ax2.set_ylabel(r'frac. diff.')
                    ax2.set_ylabel(r'ratio')
                    ax2.set_xlabel(title, fontsize = 30)
                    plt.subplots_adjust(hspace=0.0)
                    #plt.tight_layout(h_pad=0.0)
                    outfile=self.outdir+"/"+title.replace(" ","_")
                    self._saveFigure(figure,outfile,self.format)
            self._generatePlotfile(self.outdir,title,'plottemplate_scale.py')

        if self.html:
            self._WriteHTML(self.outdir,self.hlist[0].horder,self.format)

        if self.show:
            import webbrowser
            webbrowser.open(self.outdir+"/index.html")

        
        
# -----------------------------------------------------------------------------

def usage():
    print """\
Usage: histtools [OPTION...] [FILEs]
Reweight events
  -o, --output='name.hist'  Output name

  --print               Print histograms
  --add                 Add histograms
  --average             Average histograms

Other options:
  -h, --help                show this help message
"""

class Params:
    DEFAULT_ACTION = 'print_hists'

    def get_ofile(self):
        if self.oname == '-':
            return sys.stdout
        elif self.oname[-3:] == '.gz':
            return gzip.open(self.oname, 'wb')
        else:
            return open(self.oname, 'wb')

    def print_hists(self):
        f = self.get_ofile()
        for name in self.inames:
            h = RawHist.openfile(name)
            f.write(str(h))
        f.close()

    def add_hists(self):
        h = RawHist.summed(map(RawHist.openfile, self.inames))
        f = self.get_ofile()
        f.write(str(h))
        f.close()

    def avg_hists(self):
        h = RawHist.averaged(map(RawHist.openfile, self.inames))
        f = self.get_ofile()
        f.write(str(h))
        f.close()

    def __init__(self):
        try:
            opts, args = getopt.getopt(sys.argv[1:], "o:h",
                                 ["output=", "add", "average", "print", "help"])
        except getopt.GetoptError, err:
            print str(err)
            usage()
            sys.exit(2)

        self.oname = '-'
        self.action = []

        for op, oparg in opts:
            if op in ("-h", "--help"):
                usage()
                sys.exit()
            elif op in ("--add"):
                self.action.append('add_hists')
            elif op in ("--average"):
                self.action.append('avg_hists')
            elif op in ("--print"):
                self.action.append('print_hists')
            elif op in ("-o", "--oname"):
                self.oname = oparg
            else:
                assert False, "unhandled option"

        if len(self.action) == 0:
            self.action.append(Params.DEFAULT_ACTION)
        elif len(self.action) > 1:
            print "Mutually exclusive options: ", self.action
            usage()
            sys.exit(2)

        self.action = getattr(self, self.action[0])

        if len(args) >= 1:
            self.inames = args
        else:
            print "Error: missing input files"
            usage()
            sys.exit(2)

if __name__ == '__main__':
    import getopt
    import sys
    import operator
    param = Params()
    param.action()

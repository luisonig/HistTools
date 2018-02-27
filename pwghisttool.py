#! /usr/bin/env python

import sys, os
import copy
import re
import logging
import cStringIO
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Iterable
from jinja2 import Environment, FileSystemLoader, Template

__all__ = ['Alphas', 'PwgHist','PwgPlot']


class Alphas:
    def __init__(self, nf, lambda5MSB=None):
        self.nf = nf
        self.beta0 = 1. / 12. / np.pi * (33. - 2. * self.nf)
        self.beta1 = 1. / 24. / np.pi / np.pi * (153. - 19. * self.nf)
        if lambda5MSB:
            self.lambda5MSB = lambda5MSB
        else:
            self.lambda5MSB = 0.22624261289251604

    def __call__(self, scale):
        as_at_scale = self.runalphas(scale)
        return as_at_scale

    def runalphas(self, q):
        xlam = self.lambda5MSB
        xlq = 2 * np.log(q / xlam)
        xllq = np.log(xlq)
        b = self.beta0
        bp = self.beta1 / self.beta0
        if self.nf == 5:
            return 1 / (b * xlq) - bp / (b * xlq) ** 2 * xllq
        else:
            raise ValueError('Alpha_s running not implemented for this value of nf')


class PwgHist:
    def __init__(self, file=None):
        self.horder = []  # keep information about the order of the hists in the file
        self.hdata = {}   # dictionary with histograms
        self.name  = ''
        if file:
            self.name = file.name
            self.fromfile(file)

    def fromfile(self, file):

        hdata = {}

        while True:

            line = file.readline()

            # if reach and of file
            if line == '':
                break

            # if title line found
            if re.search('#', line) != None:
                histname = re.split("index", line)[0]
                histname = re.sub("#"," ",histname)
                histname = histname.lstrip()
                histname = histname.rstrip()
                self.horder.append(histname)
                sio = cStringIO.StringIO()

                while True:
                    last_pos = file.tell()
                    histoline = file.readline()
                    if re.search("#", histoline) != None or not histoline:
                        break

                    if len(histoline) > 2:
                        sio.write(histoline.replace('D', 'E'))

                sio.seek(0)
                self.hdata[histname] = np.loadtxt(sio, unpack=True, ndmin=2)
                file.seek(last_pos)

        file.close()

    def clone(self):
        new = copy.deepcopy(self)
        return new

    def __str__(self, sort=False):
        sio = cStringIO.StringIO()
        if sort:
            horder = sorted(self.horder)
        else:
            horder = self.horder
        for index, histname in enumerate(horder):
            data = self.hdata[histname]
            sio.write("#---\n# %s index %s\n" % (histname,index))
            # area = sum((data[1]-data[0])*data[2])
            # sio.write("## Area: %.15e\n" % area)
            # sio.write("## Num bins: %d\n" % len(data[0]))
            sio.write("#  xlow   xhigh     val     err\n")
            np.savetxt(sio, np.transpose(data), fmt='%.9E')
            sio.write("#---\n\n\n")
        return sio.getvalue()

    def append(self, histname, histo):
        self.horder.append(histname)
        self.hdata[histname]=np.transpose(histo)
    
    def write(self, file, sort=False):
        sio = cStringIO.StringIO()
        if sort:
            horder = sorted(self.horder)
        else:
            horder = self.horder
        for index, histname in enumerate(horder):
            data = self.hdata[histname]
            file.write("# %s index %s\n" % (histname,index))
            np.savetxt(file, np.transpose(data), fmt='%.9E')
            file.write("\n\n")
        file.close()

    def __add__(self, other):
        new = self.clone()
        for hn in self.hdata.keys():
            assert (self.hdata[hn][:2] == other.hdata[hn][:2]).all()
            (y0, e0) = self.hdata[hn][2:]
            (y1, e1) = other.hdata[hn][2:]
            new.hdata[hn][2] = y0 + y1
            new.hdata[hn][3] = np.sqrt(e0 * e0 + e1 * e1)
        return new

    def scalevariation_lo(self, Q, mu, aspow):
        new = self.clone()
        a = Alphas(5.)
        for hn in self.hdata.keys():
            (ylo, errlo) = self.hdata[hn][2:]
            new.hdata[hn][2] = ylo * (a(mu) / a(Q)) ** aspow
            new.hdata[hn][3] = errlo * (a(mu) / a(Q)) ** aspow
        return new

    def scalevariation_nlo(self, hist_lo, Q, mu, aspow):
        new = self.clone()
        a = Alphas(5.)
        twopi = 2. * np.pi
        as2pi = a(Q) / twopi
        K = twopi * aspow * a.beta0 * np.log((mu * mu) / (Q * Q))
        for hn in self.hdata.keys():
            assert (hist_lo.hdata[hn][:2] == self.hdata[hn][:2]).all()
            (ylo, errlo) = hist_lo.hdata[hn][2:]
            (ynlo, errnlo) = self.hdata[hn][2:]
            nlocoeff = (ynlo - ylo) / as2pi ** (aspow + 1) + K * ylo / as2pi ** aspow
            new.hdata[hn][2] = ylo * (a(mu) / a(Q)) ** aspow + nlocoeff * (a(mu) / twopi) ** (aspow + 1)
            new.hdata[hn][3] = np.sqrt(
                (errlo * (a(mu) / a(Q)) ** aspow * (1. + K * a(mu) / twopi - a(mu) / a(Q))) ** 2 +
                (errnlo * (a(mu) / a(Q)) ** (aspow + 1)) ** 2)
        return new


class PwgPlot:
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
                    assert isinstance(i, PwgHist)
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
                    (xl,xh,y,erry)=data
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
            self._generatePlotfile(self.outdir,title)

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
        np.set_printoptions(edgeitems=4,linewidth=100,threshold=np.nan)
        self.plotdata.append(template.render(filename=filename,
                                             title=title,
                                             data=np.array2string(np.transpose(data), separator=',', prefix='                    ')))
    
    def _generatePlotfile(self, outdir, title):
        path=os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(path))
        plotfile = open(outdir+'/'+title+'-source.py','w')
        template = env.get_template('plottemplate.py')
        data = ''
        #print self.plotdata
        for dataset in self.plotdata:
            data += dataset+"\n\n"
        plotfile.write(template.render(title=title,data=data))
        del self.plotdata[:]

        
    
if __name__ == '__main__':
    ## Argument parser
    from argparse import ArgumentParser, FileType

    parser = ArgumentParser(description='Prints histogram "histo"')

    parser.add_argument('infile', nargs=1, type=FileType('r'),
                        default=sys.stdin, metavar='histo',
                        help='A POWHEG-BOX histogram file.')

    args = parser.parse_args()
    file = args.infile

    hist = PwgHist(file[0])

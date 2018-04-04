#! /usr/bin/env python

import sys,os,logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Iterable

class Plot:

    def __init__(self,title=None):
        self.title = ''
        self.hdata = {}
        self.hlist = []
        if title:
            self.title = title

    def addData(self,name,histo):
        self.hlist.append(name)
        self.hdata[name]=histo

    def plot(self,outfile,format='png'):

        if len(self.hlist) < 8:
            color_cycle = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        else:
            colormap = plt.cm.rainbow
            color_cycle = [colormap(i) for i in np.linspace(0, 0.9, len(self.hlist))]

        font = {'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 18}
        plt.rc('font', **font)

        # Now make plots:
        figure=plt.figure(figsize = (10,10), dpi=100)
        ax1=plt.subplot2grid((5,4),(0,0),colspan=4,rowspan=4)
        ax2=plt.subplot2grid((5,4),(4,0),colspan=4,rowspan=1,sharex=ax1)
        ax1.grid(True,which='both', axis='both',alpha=0.2)
        ax2.grid(True,which='both', axis='both',alpha=0.2)
        y_min=sys.float_info.max
        y_max=-y_min
        numFound=0
        for i in range(0,len(self.hlist)):
            numFound+=1
            print("Found plot "+self.title.rstrip("\n")+" in "+self.hlist[i].rstrip("\n"))
            data=self.hdata[self.hlist[i]][self.title]
            (xl,xh,y,erry,errya,yup,erryu,erryua,ydown,errydown,errydowna)=data
            # print data
            if numFound == 1:
                valueRef=y
                errorRef=erry
            legendEntry=self.hlist[i].rstrip("\n")
            xval = np.transpose([xl,xh]).flatten()
            ax1.plot(xval, np.transpose([y,y]).flatten(),linewidth=0.7,label=legendEntry, c=color_cycle[i])
            ax1.errorbar(0.5*(xl+xh), y, yerr = erry , fmt="none", capsize=1, elinewidth=0.3, ecolor=color_cycle[i])
            ax1.fill_between(xval, np.transpose([ydown, ydown]).flatten(), np.transpose([yup, yup]).flatten(), color=color_cycle[i], alpha=0.2, lw=0.7)
            ax1.plot(xval, np.transpose([ydown, ydown]).flatten(), color=color_cycle[i], lw=0.7)
            ax1.plot(xval, np.transpose([yup, yup]).flatten(), color=color_cycle[i], lw=0.7)

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
                sc_up_ratio = [a/b if b!=0.0 else 0.0  for (a,b) in zip(yup,valueRef)]
                sc_down_ratio = [a/b if b!=0.0 else 0.0  for (a,b) in zip(ydown,valueRef)]
            else:
                value_ratio = (y/valueRef) if valueRef!=0.0 else 0.0
                error_ratio = (erry/valueRef) if valueRef!=0.0 else 0.0
                sc_up_ratio = (yup/valueRef) if valueRef!=0.0 else 0.0
                sc_down_ratio = (ydown/valueRef) if valueRef!=0.0 else 0.0

            ax2.plot(np.transpose([xl,xh]).flatten(),np.transpose([value_ratio,value_ratio]).flatten(),lw=0.7,c=color_cycle[i])
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
            ax2.set_xlabel(self.title, fontsize = 30)
            plt.subplots_adjust(hspace=0.0)
            #plt.tight_layout(h_pad=0.0)
            #outfile=self.outdir+"/"+self.title.replace(" ","_")
            self._saveFigure(figure,outfile,format)

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


if __name__ == '__main__':

    plot = Plot('{{ title }}')

{{ data }}
    
    plot.plot('{{ title }}')

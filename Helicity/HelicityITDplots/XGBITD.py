#!/usr/bin/env python3

import Plot
import numpy as np
#matplotlib inline
import matplotlib as mpl
import matplotlib.patches as mpatches

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

def loadData(filename):
    with open(filename, 'r') as File:
        Data = [tuple(float(x) for x in line.split()) for line in File
                if not line.startswith("#")]
        return tuple(zip(*Data))

# Set some style-related things.
Plot.Plot.init()

# Create a plot object with figure size.
Plotter = Plot.Plot((5, 4)); # If no argument, default is (4, 3)
Plotter.newPlot();
Plotter.setXRange(0.0, 15.01);
Plotter.setYRange(0.0, 0.045);

Plotter.Plot.set_ylabel(r'$\mathit{\Delta \mathcal{I}_g(\omega, \mu)}$', fontsize =14)
Plotter.Plot.set_xlabel(r'$\mathit{\omega}$', fontsize =14)
#Plotter.setXLabel(r"$t_1 - t_2 /2$")
Plotter.Plot.xaxis.set_label_coords(0.5, -0.08)
Plotter.Plot.yaxis.set_label_coords(-0.12, 0.5)
Plotter.Plot.yaxis.set_ticks(np.arange(0.0, 0.041, 0.01))
Plotter.Plot.yaxis.set_ticklabels([ 0.0, 0.010, 0.020, 0.030, 0.040], fontsize=10)

Plotter.Plot.xaxis.set_ticks(np.arange(0.0, 15.01, 2))
Plotter.Plot.xaxis.set_ticklabels([0, 2, 4, 6, 8, 10, 12, 14], fontsize=10)


#Plotter.Plot.text(10.9, 0.000018, r"$\mathit{m_\pi=0.278}$ GeV, $\mathit{\xi=3a}$", fontsize=18)
#Plotter.Plot.text(2.8, 1.1, r"$\mathit{\tau=0.0}$", fontsize=14)

#Plotter.Plot.fill_between(*(loadData("JAM-ITD_2negband.txt")),
#                          facecolor="royalblue", alpha=0.6, linewidth=0)
#
#
#Plotter.Plot.fill_between(*(loadData("JAM-pos-band.txt")),
#                          facecolor="orange", alpha=0.4, linewidth=0)
                          
                          
#Plotter.Plot.fill_between(*(loadData("NNPDF_band.txt")),
#                          facecolor="orchid", alpha=0.5, linewidth=0)
#
#Plotter.Plot.fill_between(*(loadData("lattice_pseudoITD_band.txt")),
#                          facecolor="crimson", alpha=0.4, linewidth=0)
                          
Plotter.Plot.fill_between(*(loadData("ITD_XGB_band.txt")),
                          facecolor="gray", alpha=0.55, linewidth=0)


#Plotter.Plot.fill_between(*(loadData("Pheno_JAM-NNPDF/NNPDF_pol_new_tech/NNPDF-band.txt")),
#                          facecolor="crimson", alpha=0.5, linewidth=0)








#p0 = Plotter.Plot.fill(0,0,'royalblue', alpha=0.6, linewidth=0)
#
#p1 = Plotter.Plot.fill(0,0,'orange', alpha=0.4, linewidth=0)

#p2 = Plotter.Plot.fill(0,0,'orchid', alpha=0.5, linewidth=0)
#
#p3 = Plotter.Plot.fill(0,0,'crimson', alpha=0.4, linewidth=0)

p4 = Plotter.Plot.fill(0,0,'gray', alpha=0.55, linewidth=0)


#platass_4 = Plotter.Plot.fill(0,0,'g', alpha=0.5, linewidth=0.0)


#p00 = Plotter.plotLine(loadData("JAM-ITD_2negline.txt"), color="royalblue", linewidth=1.5, alpha=1.0, label=" legend");
#
#
#
#p11 = Plotter.plotLine(loadData("JAM-pos-line.txt"), color="orange", linewidth=1.5, alpha=1.0, label=" legend");



#p22 = Plotter.plotLine(loadData("NNPDF_line.txt"), color="orchid", linewidth=1.5, alpha=1.0, label=" legend");
#
#p33 = Plotter.plotLine(loadData("lattice_pseudoITD_line.txt"), color="crimson", linewidth=1.5, alpha=1.0, label=" legend");

p44 = Plotter.plotLine(loadData("ITD_XGB_line.txt"), color="gray", linewidth=1.5, alpha=1.0, label=" legend");
#
#p1latass_4 = Plotter.plotLine(loadData("w1.0plots/p1z6line.txt"), color="g", linewidth=1.0, alpha=0.9, label=" legend");






#x,y,yerr= loadData("all_latt_res.txt")
#x0,y0,yerr0= loadData("p1sub_plot.txt")
#x1,y1,yerr1= loadData("p2sub_plot.txt")
#x2,y2,yerr2= loadData("p3sub_plot.txt")
#x3,y3,yerr3= loadData("p4sub_plot.txt")
#x4,y4,yerr4= loadData("p5sub_plot.txt")
#x5,y5,yerr5= loadData("p6sub_plot.txt")
#x6,y6,yerr6= loadData("p6.txt")



#p= Plotter.Plot.errorbar(x,y,yerr,ecolor='royalblue',marker='D',ls='', capsize=3.0,markeredgewidth=0.9,markeredgecolor = "none", color="royalblue",markersize = 5.0, elinewidth = 0.7,alpha=1.0)

#p1= Plotter.Plot.errorbar(x0,y0,yerr0,ecolor='gray',marker='o',ls='', capsize=3.0,markeredgewidth=0.9,markeredgecolor = "none", color="gray",markersize = 5.0, elinewidth = 0.7,alpha=1.0)
##
#p2= Plotter.Plot.errorbar(x1,y1,yerr1,ecolor='maroon',marker='s',ls='', capsize=3.0,markeredgewidth=0.9,markeredgecolor = "none", color="maroon",markersize = 5.0, elinewidth = 0.7,alpha=1.0)
##
#p3= Plotter.Plot.errorbar(x2,y2,yerr2,ecolor='orange',marker='v',ls='', capsize=3.0,markeredgewidth=0.9,markeredgecolor = "none", color="orange",markersize = 5.0, elinewidth = 0.7,alpha=1.0)
#
#p4= Plotter.Plot.errorbar(x3,y3,yerr3,ecolor='g',marker='*',ls='', capsize=3.0,markeredgewidth=0.9,markeredgecolor = "none", color="g",markersize = 7.0, elinewidth = 0.7,alpha=1.0)
#
#p5= Plotter.Plot.errorbar(x4,y4,yerr4,ecolor='royalblue',marker='^',ls='', capsize=3.0,markeredgewidth=0.9,markeredgecolor = "none", color="royalblue",markersize = 5.0, elinewidth = 0.7,alpha=1.0)
#
#p6= Plotter.Plot.errorbar(x5,y5,yerr5,ecolor='magenta',marker='D',ls='', capsize=3.0,markeredgewidth=0.9,markeredgecolor = "none", color="magenta",markersize = 5.0, elinewidth = 0.7,alpha=1.0)

#
#Plotter.makeLegends(loc= (0.05, 0.01),prop={'size':15.5}); # plot legend position argument

#Plotter.Plot.legend([p1,p2,p3,p4,p5,p6,(p2x[-1], p1x[0]),(p22[-1], p111[0])], [r' $\mathit{p=0.41}$ GeV', r' $\mathit{p=0.82}$ GeV', r' $\mathit{p=1.23}$ GeV', r' $\mathit{p=1.64}$ GeV', r' $\mathit{p=2.05}$ GeV', r' $\mathit{p=2.46}$ GeV', r' NNPDF $\mathit{\widetilde{I}_p}$', r' Fit-2'], numpoints=1,prop={'size':11.0},loc= (0.68, 0.46))

Plotter.Plot.legend([(p4[-1], p44[0])], [ r' $\mathit{\Delta \mathcal{I}_g(\omega, \mu)}$ (XGB)'], numpoints=1,prop={'size':12.0},loc= (0.55, 0.8))

#Plotter.Plot.legend([p2,p3,p4,p5,p6], [r' $\mathit{p=0.82}$ GeV', r' $\mathit{p=1.23}$ GeV', r' $\mathit{p=1.64}$ GeV', r' $\mathit{p=2.05}$ GeV', r' $\mathit{p=2.46}$ GeV'], numpoints=1,prop={'size':14.0},loc= (0.9, 0.0))





Plotter.save("DeltaIg-XGB.pdf");

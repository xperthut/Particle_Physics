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
Plotter.setXRange(0.0, 14.01);
Plotter.setYRange(0.0, 0.055);

Plotter.Plot.set_ylabel(r'$\mathit{\Delta \mathcal{I}_g(\omega, \mu^2)}$', fontsize =18)
Plotter.Plot.set_xlabel(r'$\mathit{\omega}$', fontsize =20)
#Plotter.setXLabel(r"$t_1 - t_2 /2$")
Plotter.Plot.xaxis.set_label_coords(0.5, -0.08)
Plotter.Plot.yaxis.set_label_coords(-0.12, 0.5)
Plotter.Plot.yaxis.set_ticks(np.arange(0.0, 0.051, 0.01))
Plotter.Plot.yaxis.set_ticklabels([ 0.0, 0.010, 0.020, 0.030, 0.040, 0.050], fontsize=10)

Plotter.Plot.xaxis.set_ticks(np.arange(0.0, 15.01, 2))
Plotter.Plot.xaxis.set_ticklabels([0, 2, 4, 6, 8, 10, 12, 14], fontsize=12)


#Plotter.Plot.text(10.9, 0.000018, r"$\mathit{m_\pi=0.278}$ GeV, $\mathit{\xi=3a}$", fontsize=18)
#Plotter.Plot.text(2.8, 1.1, r"$\mathit{\tau=0.0}$", fontsize=14)


Plotter.Plot.fill_between(*(loadData("NNPDFpol_data/data_NNPDF_ITD_band.txt")),
                          facecolor="gray", alpha=0.55, linewidth=0)
#Plotter.Plot.fill_between(*(loadData("ITD_XGB_band.txt")),
#                          facecolor="gray", alpha=0.55, linewidth=0)
Plotter.Plot.fill_between(*(loadData("ITD_RF_band_sys.txt")),
                          facecolor="royalblue", alpha=0.4, linewidth=0)
                          
Plotter.Plot.fill_between(*(loadData("ITD_RF_band.txt")),
                          facecolor="c", alpha=0.55, linewidth=0)
                          
                          



#Plotter.Plot.fill_between(*(loadData("Pheno_JAM-NNPDF/NNPDF_pol_new_tech/NNPDF-band.txt")),
#                          facecolor="crimson", alpha=0.5, linewidth=0)








#p0 = Plotter.Plot.fill(0,0,'royalblue', alpha=0.6, linewidth=0)
#
#p1 = Plotter.Plot.fill(0,0,'orange', alpha=0.4, linewidth=0)

#p2 = Plotter.Plot.fill(0,0,'orchid', alpha=0.5, linewidth=0)
#
#p3 = Plotter.Plot.fill(0,0,'crimson', alpha=0.4, linewidth=0)
p4 = Plotter.Plot.fill(0,0,'gray', alpha=0.4, linewidth=0)
#p1 = Plotter.Plot.fill(0,0,'gray', alpha=0.4, linewidth=0)
p2 = Plotter.Plot.fill(0,0,'royalblue', alpha=0.4, linewidth=0)
p3 = Plotter.Plot.fill(0,0,'c', alpha=0.4, linewidth=0)



#platass_4 = Plotter.Plot.fill(0,0,'g', alpha=0.5, linewidth=0.0)


#p00 = Plotter.plotLine(loadData("JAM-ITD_2negline.txt"), color="royalblue", linewidth=1.5, alpha=1.0, label=" legend");
#
#
#
#p11 = Plotter.plotLine(loadData("JAM-pos-line.txt"), color="orange", linewidth=1.5, alpha=1.0, label=" legend");



#p22 = Plotter.plotLine(loadData("NNPDF_line.txt"), color="orchid", linewidth=1.5, alpha=1.0, label=" legend");
#
#p33 = Plotter.plotLine(loadData("lattice_pseudoITD_line.txt"), color="crimson", linewidth=1.5, alpha=1.0, label=" legend");
p44 = Plotter.plotLine(loadData("NNPDFpol_data/data_NNPDF_ITD_line.txt"), color="crimson", linewidth=1.5, alpha=1.0, label=" legend");

#p11 = Plotter.plotLine(loadData("ITD_XGB_line.txt"), color="gray", linewidth=1.5, alpha=1.0, label=" legend");
#
p22 = Plotter.plotLine(loadData("ITD_RF_line.txt"), color="royalblue", linewidth=1.5, alpha=1.0, label=" legend");

p33 = Plotter.plotLine(loadData("ITD_RF_line.txt"), color="c", linewidth=1.5, alpha=1.0, label=" legend");


#


Plotter.Plot.legend([(p3[-1], p33[0]), (p2[-1], p22[0]), (p4[-1], p44[0])], [ r' RF (stat.)', r' RF (stat.$+$sys.)', r' NNPDF'], numpoints=1,prop={'size':14.0},loc= (0.65, 0.62))

#Plotter.Plot.legend([p2,p3,p4,p5,p6], [r' $\mathit{p=0.82}$ GeV', r' $\mathit{p=1.23}$ GeV', r' $\mathit{p=1.64}$ GeV', r' $\mathit{p=2.05}$ GeV', r' $\mathit{p=2.46}$ GeV'], numpoints=1,prop={'size':14.0},loc= (0.9, 0.0))





Plotter.save("DeltaIg-ML-NNPDF_SYS.pdf");

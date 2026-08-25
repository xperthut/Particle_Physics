import matplotlib
# import matplotlib.pyplot as Plt
import matplotlib.figure
import matplotlib.font_manager as Fm
import matplotlib.backends.backend_pdf as PltPDF

class Plot(object):
    def __init__(self, fig_size=(4,3)):
        self.DefaultPara = {"linestyle": "none",
                            "markeredgewidth": 0,
                            "markersize": 2.5,
                            "linewidth": 0.5,
                            "marker": 'o'}
        self.DefaultParaErrorbar = dict()

        self.DefaultLinePara = {"linestyle": "-",
                                "linewidth": 1,
                                "markersize": 0}

        self.XRange = None
        self.YRange = None
        self.LogScale = False
        self.HLine = None
        self.Fig = None
        self.Plot = None        # This represents a subplot.
        self.ErrorCapSize = 0
        # Set this to a float to shift x value further more a little
        # bit for every plot.
        self.AutoShiftX = False
        self.__PlotCount = 0
        self.__newFig(fig_size)

    @staticmethod
    def init():
        matplotlib.rc("legend", numpoints=1, labelspacing=0.2, handletextpad=0,
                      columnspacing=0.2, frameon=False, fontsize=7)
        matplotlib.rc("mathtext", default="regular")
        matplotlib.rc("figure.subplot", hspace=0.3)
        matplotlib.rc("lines", marker="o", markeredgewidth=0, markersize=3,
                      dash_joinstyle="round",
                      solid_joinstyle="round")
        matplotlib.rc("savefig", dpi=128, transparent=True)
        matplotlib.rc("figure", dpi=128)
        matplotlib.rc("grid", alpha=0.3)
        matplotlib.rc("path", simplify=True, snap=True)
        matplotlib.rc("font", family="serif")
        matplotlib.rc("text", hinting="either")

        Font = {
            # 'family' : 'Helvetica',
            'size'   : 8,}
        matplotlib.rc('font', **Font)

    def hLine(self, y=0):
        """Draw a horizontal line at this y number.
        """
        self.Plot.axhline(y, linestyle='-', color="black")
        return self

    def setTitle(self, title):
        self.Fig.suptitle(title)
        return self

    def setSubTitle(self, title):
        self.Plot.set_title(title)
        return self

    def setXLabel(self, label):
        self.Plot.set_xlabel(label)
        return self

    def setXRange(self, x1, x2):
        self.Plot.set_xlim(x1, x2)
        return self

    def setYRange(self, x1, x2):
        self.Plot.set_ylim(x1, x2)
        return self

    def setYLabel(self, label):
        self.Plot.set_ylabel(label)
        return self

    def setErrorCap(self, size=0):
        """Zero means no errorbar cap.  Otherwise size should be a number
        greater than 0.  Unit is point.
        """
        self.ErrorCapSize = size
        self.DefaultParaErrorbar["capsize"] = size
        self.DefaultParaErrorbar["capthick"] = 1
        return self

    def makeLegends(self, loc=0, **kwargs):
        # if not ("prop" in kwargs):
        #     kwargs["prop"] = {"size": 10}

        self.Plot.legend(loc=loc, **kwargs)
        return self

    def makeGrid(self):
        self.Plot.grid()
        return self

    def save(self, filename, dpi=None, transparency=True):
        self.Fig.savefig(filename, bbox_inches='tight', dpi=dpi,
                         transparent=transparency)
        return self

    def plotLine(self, data, **kwargs):
        def fillArgs(kwargs, fill_dict):
            Args = {}
            Args.update(fill_dict)
            Args.update(kwargs)
            return Args
        Args = fillArgs(kwargs, self.DefaultLinePara)
        return self.Plot.plot(data[0], data[1], **Args)

    def setLogScale(self, logscale="y"):
        if "x" in logscale:
            self.Plot.set_xscale("log")
        if "y" in logscale:
            self.Plot.set_yscale("log")
        return self

    def plot(self, data, **kwargs):
        def fillArgs(kwargs, fill_dict):
            Args = {}
            Args.update(fill_dict)
            Args.update(kwargs)
            return Args

        if self.Plot is None:
            self.newPlot()

        if len(data) == 3:
            HaveErr = True
        elif len(data) == 2:
            HaveErr = False
        else:
            raise RuntimeError("Data needs to be length 2 or 3 in order to plot.")

        # Shift x
        PlotData = []
        if self.__PlotCount > 0 and self.AutoShiftX:
            PlotData.append([x + float(self.__PlotCount) * self.AutoShiftX
                               for x in data[0]])
        else:
            PlotData.append(data[0])
        for dim in data[1:]:
            PlotData.append(dim)

        Para = dict()
        Para.update(self.DefaultPara)
        if HaveErr:
            Para.update(self.DefaultParaErrorbar)
            Para["capsize"] = self.ErrorCapSize
        elif "capsize" in self.DefaultPara:
            del self.DefaultPara["capsize"]

        self.__PlotCount += 1
        Args = fillArgs(kwargs, Para)
        if HaveErr:
            # Has error
            (Stuff1, Cap, Stuff2) = self.Plot.errorbar(
                PlotData[0], PlotData[1], yerr=PlotData[2], **Args)
            if self.ErrorCapSize > 0:
                for cap in Cap:
                    cap.set_markeredgewidth(1)
            return (Stuff1, Cap, Stuff2)
        else:
            return self.Plot.plot(PlotData[0], PlotData[1], **Args)

    def __newFig(self, size=(4,3)):
        Fig = matplotlib.figure.Figure(figsize=size) # Letter paper
        matplotlib.backends.backend_agg.FigureCanvasAgg(Fig) # ???
        Fig.subplots_adjust(left=0.12, top=0.9, bottom=0.15, right=0.98)
        self.Fig = Fig
        return self

    def newPlot(self, *subplot_spec):
        if len(subplot_spec) == 0:
            Ax = self.Fig.add_subplot(111)
        else:
            Ax = self.Fig.add_subplot(*subplot_spec)
        self.Plot = Ax
        self.__PlotCount = 0
        return self

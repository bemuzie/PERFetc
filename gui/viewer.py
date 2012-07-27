#!/usr/bin/env python
#-*-coding: utf-8 -*-
import sys
import matplotlib
matplotlib.use('Qt4Agg')
#matplotlib.rcParams["backend.qt4"]="PySide"
import pylab
import numpy as np

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import SIGNAL,SLOT
from scipy import stats


def gammapdf(x,*args):
    return stats.gamma.pdf(x,*args)
def gammacdf(x,*args):
    return stats.gamma.cdf(x,*args)

class MyGraf(FigureCanvas):
    def __init__(self,parent=None, width=5, height=4, dpi=100):
        fig=Figure(figsize=(width,height),dpi=dpi)
        self.axes=fig.add_subplot(111)
        self.axes.hold(False)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
            QtGui.QSizePolicy.Expanding,
            QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    def compute_graph(self,function,ox,f,s,t):
        x=np.arange(0.0001,ox,0.1)
        y=function(x,f,s,t)
        y/=np.sum(y)
        self.axes.plot(x,y)
        self.draw()



class MyApp(QtGui.QWidget):
    def __init__(self,*args):
        QtGui.QWidget.__init__(self,*args)
        self.setWindowTitle(u"Гамма распределение")

        self.xaxis=QtGui.QSpinBox()
        self.firstp=QtGui.QDoubleSpinBox()
        self.secondp=QtGui.QDoubleSpinBox()
        self.thirdp=QtGui.QDoubleSpinBox()
        startbutton=QtGui.QPushButton(u'Построить график')

        self.xaxis.setValue(10)

        for sb in [self.firstp,self.secondp,self.thirdp]:
            sb.setValue(1)
            sb.setSingleStep(0.1)
            self.connect(sb,SIGNAL("valueChanged(double)"),self.drawgraph)

        self.connect(startbutton,SIGNAL("clicked()"),self.drawgraph)
        self.pdfwidget=MyGraf()
        self.cdfwidget=MyGraf()

        layout=QtGui.QHBoxLayout()
        layout.addWidget(self.pdfwidget)
        layout.addWidget(self.cdfwidget)
        layout.addWidget(self.xaxis)
        layout.addWidget(self.firstp)
        layout.addWidget(self.secondp)
        layout.addWidget(self.thirdp)
        layout.addWidget(startbutton)
        self.setLayout(layout)

    def drawgraph(self):
        x=self.xaxis.value()
        f=self.firstp.value()
        s=self.secondp.value()
        t=self.thirdp.value()


        self.pdfwidget.compute_graph(gammapdf,x,f,s,t)
        self.cdfwidget.compute_graph(gammacdf,x,f,s,t)



if __name__ == '__main__':
    app=QtGui.QApplication(sys.argv)
    ns=MyApp()
    ns.show()
    sys.exit(app.exec_())
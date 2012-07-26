#!/usr/bin/env python
#-*-coding: utf-8 -*-
import sys
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams["backend.qt4"]="PySide"
import pylab

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide import QtCore, QtGui
from PySide.QtCore import SIGNAL,SLOT

class NumSelector(QtGui.QWidget):
    def __init__(self,*args):
        QtGui.QWidget.__init__(self,*args)
        self.setWindowTitle(u"Введите число")
        spinbox=QtGui.QSpinBox()
        slider=QtGui.QSlider(QtCore.Qt.Horizontal)
        spinbox.setRange(0,138)
        slider.setRange(0,138)
        self.connect(spinbox,SIGNAL("valueChanged(int)"),slider,SLOT("setValue(int)"))
        self.connect(slider,SIGNAL("valueChanged(int)"),spinbox,SLOT("setValue(int)"))
        self.connect(slider,SIGNAL("valueChanged(int)"),self.console_log)
        spinbox.setValue(27)
        layout=QtGui.QHBoxLayout()
        layout.addWidget(spinbox)
        layout.addWidget(slider)
        self.setLayout(layout)
    def console_log(self,i):
        print i



if __name__ == '__main__':
    app=QtGui.QApplication(sys.argv)
    ns=NumSelector()
    ns.show()
    sys.exit(app.exec_())
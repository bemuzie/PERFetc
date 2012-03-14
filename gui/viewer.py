__author__ = 'ct'


import sys
from PySide.QtCore import *
from PySide.QtGui import *


# Create a Qt application
app = QApplication(sys.argv)
# Create a Label and show it
label = QLabel("<font background-color=rgb(204, 204, 204) color=red size=40 >Hello World</font>")
label.show()
# Enter Qt application main loop
app.exec_()
sys.exit()


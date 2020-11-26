# -*- coding: utf-8 -*-

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import mainwindow
import pupil_apriltags


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = mainwindow.MainWindow()
    main_window.show()

sys.exit(app.exec_())
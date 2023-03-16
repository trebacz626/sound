import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector, Button
import matplotlib.pyplot as plt

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create the main widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        hbox = QtWidgets.QHBoxLayout(central_widget)

        # Create the left-side widget and layout
        left_widget = QtWidgets.QWidget()
        hbox.addWidget(left_widget)
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        # Create the right-side widget and layout
        right_widget = QtWidgets.QWidget()
        hbox.addWidget(right_widget)
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        # Create the plots and add them to the left-side layout
        for i in range(3):
            fig, ax = plt.subplots()
            line, = ax.plot(np.random.rand(100))
            left_layout.addWidget(FigureCanvas(fig))

        # Create the text fields and add them to the right-side layout
        for i in range(5):
            right_layout.addWidget(QtWidgets.QLineEdit())

        # Set the layout for the main widget
        central_widget.setLayout(hbox)

        # Set the window properties
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Plotting and Text Example')
        self.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())

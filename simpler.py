import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from scipy.io import wavfile
from sound_params import *


framerate, sound = wavfile.read('./sounds/song.wav')
sound = sound[:, 1].astype(np.float64)
nframes = len(sound)
duration = nframes / framerate
print(framerate)
times = np.array([i/framerate for i in range(nframes)])
frame_size = 16


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.figure1 = Figure(figsize=(5, 4), dpi=100)
        self.canvas1 = FigureCanvas(self.figure1)
        self.ax1 = self.figure1.add_subplot(111)
        self.draw_plot_sound()

        self.figure2 = Figure(figsize=(5, 4), dpi=100)
        self.canvas2 = FigureCanvas(self.figure2)
        self.ax2 = self.figure2.add_subplot(111)
        self.draw_plot_volume()

        self.figure3 = Figure(figsize=(5, 4), dpi=100)
        self.canvas3 = FigureCanvas(self.figure3)
        self.ax3 = self.figure3.add_subplot(111)
        self.draw_plot_zcr()

        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        hbox = QHBoxLayout()

        vbox1.addWidget(self.canvas1)
        vbox1.addWidget(self.canvas2)
        vbox1.addWidget(self.canvas3)


        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_plots)
        hbox.addWidget(self.reset_button)

        self.span_selector = SpanSelector(self.ax1, self.onselect, 'horizontal', useblit=True)

        hbox.addLayout(vbox1)
        hbox.addStretch(1)


        self.text_fields = []
        for i in range(5):
            label = QLabel(f'Text Field {i+1}:')
            text_field = QLineEdit()
            vbox2.addWidget(label)
            vbox2.addWidget(text_field)
            self.text_fields.append(text_field)

        hbox.addLayout(vbox2)
        central_widget = QWidget()
        central_widget.setLayout(hbox)
        self.setCentralWidget(central_widget)

    def onselect(self, xmin, xmax):

        x_min = np.searchsorted(times, xmin, side="left")
        x_max = np.searchsorted(times, xmax, side="right")

        self.draw_plot_sound(x_min, x_max)
        self.draw_plot_volume(x_min, x_max)
        self.draw_plot_zcr(x_min, x_max)


    def draw_plot_sound(self, x_min=0, x_max=sys.maxsize):
        self.ax1.clear()
        self.ax1.plot(times[x_min:x_max], sound[x_min:x_max])
        self.ax1.set_title('Sound Wave')
        self.canvas1.draw()

    def draw_plot_volume(self, x_min=0, x_max=sys.maxsize):
        self.ax2.clear()
        self.ax2.plot(times[(frame_size - 1) // 2:-frame_size // 2][x_min:x_max],
                      volume(sound, frame_size)[x_min:x_max])
        self.ax2.set_title('Volume')
        self.canvas2.draw()

    def draw_plot_zcr(self, x_min=0, x_max=sys.maxsize):
        self.ax3.clear()
        self.ax3.plot(times[(frame_size - 1) // 2:-frame_size // 2][x_min:x_max], zcr(sound, frame_size)[x_min:x_max])
        self.ax3.set_title('Zero Crossing Rate')
        self.canvas3.draw()

    def reset_plots(self):
        self.draw_plot_sound()
        self.draw_plot_volume()
        self.draw_plot_zcr()


app = QApplication([])
window = MainWindow()
window.setGeometry(100, 100, 1200, 800)
window.show()
app.exec_()

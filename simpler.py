import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector
from scipy.io import wavfile
from sound_params import *
from scipy.signal import find_peaks


framerate, sound = wavfile.read('./sounds/me.wav')
# sound = sound[:, 1].astype(np.float64)
nframes = len(sound)
duration = nframes / framerate
print(framerate)
times = np.array([i/framerate for i in range(nframes)])
fram_size_ms = 40
frame_size = framerate/(1000/fram_size_ms)

calculated_volume = efficient_volume(sound, frame_size)
calculated_zcr = efficient_zcr(sound, frame_size)
times_window = efficient_sample_time(times, frame_size)
silence = times_window[(calculated_zcr > 0.07) & (calculated_volume < 200)]


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

        self.vdr_label = QLabel(f"VDR:{vdr(sound, frame_size)}")
        # text_field = QLineEdit()
        vbox2.addWidget(self.vdr_label)
        hbox.addLayout(vbox2)
        central_widget = QWidget()
        central_widget.setLayout(hbox)
        self.setCentralWidget(central_widget)

    def onselect(self, xmin, xmax):

        x_min = np.searchsorted(times, xmin, side="left")
        x_max = np.searchsorted(times, xmax, side="right")

        self.draw_plot_sound(x_min, x_max)
        self.draw_plot_volume(xmin, xmax)
        self.draw_plot_zcr(xmin, xmax)


    def draw_plot_sound(self, x_min=0, x_max=sys.maxsize):
        self.ax1.clear()
        selected_times = times[x_min:x_max]
        selected_sounds = sound[x_min:x_max]
        for s in silence[(silence >= selected_times[0]) & (silence <= selected_times[-1])]:
            self.ax1.add_patch(Rectangle((s-fram_size_ms/1000, -10000), fram_size_ms/1000, 20000, facecolor="red"))
        self.ax1.plot(selected_times, sound[x_min:x_max])
        self.ax1.set_title('Sound Wave')
        self.ax1.set_ylim((selected_sounds.min()*0.9, selected_sounds.max()*1.1))
        self.canvas1.draw()

    def draw_plot_volume(self, xmin=0, xmax=sys.maxsize):
        self.ax2.clear()

        x_min = np.searchsorted(times_window, xmin, side="left")
        x_max = np.searchsorted(times_window, xmax, side="right")

        self.ax2.plot(times_window[x_min:x_max], calculated_volume[x_min:x_max])
        self.ax2.set_title('Volume')
        self.canvas2.draw()

    def draw_plot_zcr(self, xmin=0, xmax=sys.maxsize):
        self.ax3.clear()

        x_min = np.searchsorted(times_window, xmin, side="left")
        x_max = np.searchsorted(times_window, xmax, side="right")

        self.ax3.plot(times_window[x_min:x_max], calculated_zcr[x_min:x_max])
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

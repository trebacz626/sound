import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector
from scipy.io import wavfile
from sound_params import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


framerate, sound = wavfile.read("./sounds/me.wav")
n_frames = len(sound)
duration = n_frames / framerate
times = np.array([i / framerate for i in range(n_frames)])
frame_size_ms = 40
frame_size = framerate/(1000 / frame_size_ms)

calculated_volume = efficient_volume(sound, frame_size)
calculated_zcr = efficient_zcr(sound, frame_size)
times_window = efficient_sample_time(times, frame_size)
seconds_window = efficient_sample_time(times_window, 1000 // frame_size_ms)
silence = times_window[(calculated_zcr > 0.07) & (calculated_volume < 200)]

fundamental_freq = efficient_fundamental_frequency(sound, framerate, framerate*400//1000)
times_fundamental_frequency = efficient_sample_time(times, framerate*400//1000)
calculated_lster = lster(sound, frame_size, framerate)

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()


        self.figure1 = pg.PlotWidget(title='Sound Wave')
        self.figure2 = pg.PlotWidget(title='Volume')
        self.figure3 = pg.PlotWidget(title='Zero Crossing Rate')
        self.figure4 = pg.PlotWidget(title='Fundamental Frequency')
        self.figure5 = pg.PlotWidget(title='LSTER')

        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.figure1)
        vbox1.addWidget(self.figure2)
        vbox1.addWidget(self.figure3)

        vbox2 = QVBoxLayout()
        vbox2.addWidget(QLabel(f"VDR:{vdr(sound, frame_size)}"))
        vbox2.addWidget(self.figure4)
        vbox2.addWidget(self.figure5)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)

        central_widget = QWidget()
        central_widget.setLayout(hbox)
        self.setCentralWidget(central_widget)

        self.draw_plot_sound()
        self.draw_plot_volume()
        self.draw_plot_zcr()
        # self.draw_plot_fundamental_frequency()
        # self.draw_plot_lster()

        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_plots)
        hbox.addWidget(self.reset_button)

        self.play_button = QPushButton('Play')
        hbox.addWidget(self.play_button)

        self.span_selector = pg.LinearRegionItem([0, 100])
        self.span_selector.sigRegionChanged.connect(self.onselect)
        self.figure1.addItem(self.span_selector)

    def onselect(self):
        xmin, xmax = self.span_selector.getRegion()
        x_min = np.searchsorted(times, xmin, side="left")
        x_max = np.searchsorted(times, xmax, side="right")

        self.draw_plot_sound(x_min, x_max)
        self.draw_plot_volume(x_min, x_max)
        self.draw_plot_zcr(x_min, x_max)

    def draw_plot_sound(self, x_min=0, x_max=sys.maxsize):
        self.figure1.clear()
        selected_times = times[x_min:x_max]
        selected_sounds = sound[x_min:x_max]
        # for s in silence[(silence >= selected_times[0]) & (silence <= selected_times[-1])]:
        #     rect = pg.QtGui.QGraphicsRectItem(s - frame_size_ms / 1000, -10000, frame_size_ms / 1000, 20000)
        #     rect.setBrush(pg.mkBrush('r'))
        #     self.figure1.addItem(rect)
        self.figure1.plot(selected_times, sound[x_min:x_max])
        self.figure1.setXRange(selected_times.min(), selected_times.max())
        self.figure1.setYRange(selected_sounds.min() * 0.9, selected_sounds.max() * 1.1)

    def draw_plot_volume(self, x_min=0, x_max=sys.maxsize):
        self.figure2.clear()
        x_min = np.searchsorted(times_window, x_min, side="left")
        x_max = np.searchsorted(times_window, x_max, side="right")
        self.figure2.plot(times_window[x_min:x_max], calculated_volume[x_min:x_max])

    def draw_plot_zcr(self, x_min=0, x_max=sys.maxsize):
        self.figure3.clear()
        x_min = np.searchsorted(times_window, x_min, side="left")
        x_max = np.searchsorted(times_window, x_max, side="right")
        self.figure2.plot(times_window[x_min:x_max], calculated_zcr[x_min:x_max])

    def reset_plots(self):
        self.draw_plot_sound()
        self.draw_plot_volume()
        self.draw_plot_zcr()


app = QApplication([])
window = MainWindow()
window.setGeometry(100, 100, 1200, 800)
window.show()
app.exec_()

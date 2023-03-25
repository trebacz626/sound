import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector
from scipy.io import wavfile
from sound_params import *


framerate, sound = wavfile.read('C:\\Users\\kubak\\Documents\\Studia\\6 semestr\\AIPD\\Nagrania\\2_05.wav')
n_frames = len(sound)
duration = n_frames / framerate
print(framerate)
times = np.array([i / framerate for i in range(n_frames)])
frame_size_ms = 40
frame_size = framerate/(1000 / frame_size_ms)

calculated_volume = efficient_volume(sound, frame_size)
calculated_zcr = efficient_zcr(sound, frame_size)
times_window = efficient_sample_time(times, frame_size)
seconds_window = efficient_sample_time(times_window, 1000 // frame_size_ms)
silence = times_window[(calculated_zcr > 0.07) & (calculated_volume < 200)]

fundamental_freq = efficient_fundamental_frequency(sound, 10, frame_size)
calculated_lster = lster(sound, frame_size, framerate)


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

        self.vdr_label = QLabel(f"VDR:{vdr(sound, frame_size)}")
        vbox2.addWidget(self.vdr_label)

        self.figure4 = Figure(figsize=(5, 4), dpi=100)
        self.canvas4 = FigureCanvas(self.figure4)
        self.ax4 = self.figure4.add_subplot(111)
        self.draw_plot_fundamental_frequency()
        vbox2.addWidget(self.canvas4)

        self.figure5 = Figure(figsize=(5, 4), dpi=100)
        self.canvas5 = FigureCanvas(self.figure5)
        self.ax5 = self.figure5.add_subplot(111)
        self.draw_plot_lster()
        vbox2.addWidget(self.canvas5)

        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        hbox.addStretch(1)

        central_widget = QWidget()
        central_widget.setLayout(hbox)
        self.setCentralWidget(central_widget)

    def onselect(self, xmin, xmax):

        x_min = np.searchsorted(times, xmin, side="left")
        x_max = np.searchsorted(times, xmax, side="right")

        self.draw_plot_sound(x_min, x_max)
        self.draw_plot_volume(xmin, xmax)
        self.draw_plot_zcr(xmin, xmax)
        self.draw_plot_fundamental_frequency(xmin, xmax)
        self.draw_plot_lster(xmin, xmax)

    def draw_plot_sound(self, x_min=0, x_max=sys.maxsize):
        self.ax1.clear()
        selected_times = times[x_min:x_max]
        selected_sounds = sound[x_min:x_max]
        for s in silence[(silence >= selected_times[0]) & (silence <= selected_times[-1])]:
            self.ax1.add_patch(Rectangle((s - frame_size_ms / 1000, -10000), frame_size_ms / 1000, 20000, facecolor="red"))
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

    def draw_plot_fundamental_frequency(self, x_min=0, x_max=sys.maxsize):
        self.ax4.clear()

        min_idx = np.searchsorted(times_window, x_min, side="left")
        max_idx = np.searchsorted(times_window, x_max, side="right")

        self.ax4.plot(times_window[min_idx:max_idx], fundamental_freq[min_idx:max_idx])
        self.ax4.set_title('Fundamental Frequency')
        self.canvas4.draw()

    def draw_plot_lster(self, x_min=0, x_max=sys.maxsize):
        self.ax5.clear()

        min_idx = np.searchsorted(seconds_window, x_min, side="left")
        max_idx = np.searchsorted(seconds_window, x_max, side="right")

        self.ax5.plot(seconds_window[min_idx:max_idx], calculated_lster[min_idx:max_idx])
        self.ax5.set_title('LSTER')
        self.canvas5.draw()

    def reset_plots(self):
        self.draw_plot_sound()
        self.draw_plot_volume()
        self.draw_plot_zcr()


app = QApplication([])
window = MainWindow()
window.setGeometry(100, 100, 1200, 800)
window.show()
app.exec_()

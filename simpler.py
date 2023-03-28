import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, \
    QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector
from scipy.io import wavfile

from sound_params import *

from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import sounddevice as sd


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.loadFile("./sounds/znormalizowane/chrzaszcz_1.wav")
        self.figure1 = Figure(figsize=(5, 4), dpi=100)
        self.canvas1 = FigureCanvas(self.figure1)
        self.ax1 = self.figure1.add_subplot(111)

        self.figure2 = Figure(figsize=(5, 4), dpi=100)
        self.canvas2 = FigureCanvas(self.figure2)
        self.ax2 = self.figure2.add_subplot(111)

        self.figure3 = Figure(figsize=(5, 4), dpi=100)
        self.canvas3 = FigureCanvas(self.figure3)
        self.ax3 = self.figure3.add_subplot(111)

        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        hbox = QHBoxLayout()
        hboxbutton = QHBoxLayout()

        vbox1.addWidget(self.canvas1)
        vbox1.addWidget(self.canvas2)
        vbox1.addWidget(self.canvas3)

        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_plots)
        hboxbutton.addWidget(self.reset_button)

        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.play_audio)
        hboxbutton.addWidget(self.play_button)
        self.player = QMediaPlayer()

        self.fileOpenButton = QPushButton('Click to open File Dialog', self)
        hboxbutton.addWidget(self.fileOpenButton)
        self.fileOpenButton.clicked.connect(self.selectFile)

        vbox1.addLayout(hboxbutton)

        self.span_selector = SpanSelector(self.ax1, self.onselect, 'horizontal', useblit=True)

        self.vdr_label = QLabel(f"VDR:{vdr(self.sound, self.frame_size)}")
        vbox2.addWidget(self.vdr_label)

        self.figure4 = Figure(figsize=(5, 4), dpi=100)
        self.canvas4 = FigureCanvas(self.figure4)
        self.ax4 = self.figure4.add_subplot(111)
        vbox2.addWidget(self.canvas4)

        self.figure5 = Figure(figsize=(5, 4), dpi=100)
        self.canvas5 = FigureCanvas(self.figure5)
        self.ax5 = self.figure5.add_subplot(111)
        vbox2.addWidget(self.canvas5)

        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        hbox.addStretch(1)

        central_widget = QWidget()
        central_widget.setLayout(hbox)
        self.setCentralWidget(central_widget)
        self.onselect(0, len(self.sound))

    """
        Opens a select wav file button and sets the sound to the selected file
    """
    def selectFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.loadFile(fileName)

        self.reset_plots()

    def loadFile(self, fileName):
            print("Loading file: " + fileName)
            self.framerate, self.sound = wavfile.read(fileName)
            # self.sound /= self.sound.abs().max()
            self.n_frames = len(self.sound)
            self.duration = self.n_frames / self.framerate
            self.times = np.array([i / self.framerate for i in range(self.n_frames)])
            self.frame_size_ms = 40
            self.frame_size = self.framerate / (1000 / self.frame_size_ms)

            self.calculated_volume = efficient_volume(self.sound, self.frame_size)
            self.calculated_zcr = efficient_zcr(self.sound, self.frame_size)
            self.times_window = efficient_sample_time(self.times, self.frame_size)
            self.seconds_window = efficient_sample_time(self.times_window, 1000 // self.frame_size_ms)
            # silence_selector = (self.calculated_zcr > 0.07) & (self.calculated_volume < 800)
            # silence_selector = [silence_selector[0]] + ((silence_selector[0:-2] & silence_selector[2]) | silence_selector[1:-1]).tolist() + [silence_selector[-1]]
            #find start and end indexes of continous sections of 1s in silcene_selctor
            self.silence = find_silences(self.calculated_zcr, self.calculated_volume, self.times_window)
            # self.silence = self.times_window[silence_selector]

            self.fundamental_frequency_window = 40
            self.fundamental_freq_frame_size = self.framerate * self.fundamental_frequency_window // 1000
            self.fundamental_freq = efficient_fundamental_frequency(self.sound, self.framerate, self.fundamental_freq_frame_size)
            self.times_fundamental_frequency = efficient_sample_time(self.times, self.fundamental_freq_frame_size)
            self.calculated_lster = lster(self.sound, self.frame_size, self.framerate)
            print("File loaded")

    def play_audio(self):
        sd.play(self.sound[self.x_min:self.x_max], self.framerate)

    def stop_playing(self):
        sd.stop()

    def onselect(self, xmin, xmax):

        self.x_min = np.searchsorted(self.times, xmin, side="left")
        self.x_max = np.searchsorted(self.times, xmax, side="right")

        self.draw_plot_sound(self.x_min, self.x_max)
        self.draw_plot_volume(xmin, xmax)
        self.draw_plot_zcr(xmin, xmax)
        self.draw_plot_fundamental_frequency(xmin, xmax)
        self.draw_plot_lster(xmin, xmax)


    def draw_plot_sound(self, x_min=0, x_max=sys.maxsize):
        self.ax1.clear()
        selected_times = self.times[x_min:x_max]
        selected_sounds = self.sound[x_min:x_max]
        # for s in self.silence[(self.silence >= selected_times[0]) & (self.silence <= selected_times[-1])]:
        for start, end in self.silence:

            if end < selected_times[0]:
                continue
            if start > selected_times[-1]:
                break
            print("B",start, end)
            start = max(start, selected_times[0])
            end = min(end, selected_times[-1])
            print("A",start,end)
            self.ax1.add_patch(Rectangle((start - self.frame_size_ms / 2000, -10000), end-start+(self.frame_size_ms)/1000, 20000, facecolor="red" if start - end > 0.5 else "yellow"))
        self.ax1.plot(selected_times, selected_sounds)
        self.ax1.set_title('Sound Wave')
        self.ax1.set_ylim((selected_sounds.min()*0.9, selected_sounds.max()*1.1))
        self.canvas1.draw()



    def draw_plot_volume(self, xmin=0, xmax=sys.maxsize):
        self.ax2.clear()

        x_min = np.searchsorted(self.times_window, xmin, side="left")
        x_max = np.searchsorted(self.times_window, xmax, side="right")

        self.ax2.plot(self.times_window[x_min:x_max], self.calculated_volume[x_min:x_max])
        self.ax2.set_title('Volume')
        self.ax2.set_xlim((self.ax1.get_xlim()[0], self.ax1.get_xlim()[1]))
        self.canvas2.draw()

    def draw_plot_zcr(self, xmin=0, xmax=sys.maxsize):
        self.ax3.clear()

        x_min = np.searchsorted(self.times_window, xmin, side="left")
        x_max = np.searchsorted(self.times_window, xmax, side="right")

        self.ax3.plot(self.times_window[x_min:x_max], self.calculated_zcr[x_min:x_max])
        self.ax3.set_title('Zero Crossing Rate')
        self.ax3.set_xlim((self.ax1.get_xlim()[0], self.ax1.get_xlim()[1]))
        self.canvas3.draw()

    def draw_plot_fundamental_frequency(self, x_min=0, x_max=sys.maxsize):
        self.ax4.clear()

        min_idx = np.searchsorted(self.times_fundamental_frequency, x_min, side="left")
        max_idx = np.searchsorted(self.times_fundamental_frequency, x_max, side="right")

        self.ax4.plot(self.times_fundamental_frequency[min_idx:max_idx], self.fundamental_freq[min_idx:max_idx])
        self.ax4.set_title('Fundamental Frequency')
        self.ax4.set_xlim((self.ax1.get_xlim()[0], self.ax1.get_xlim()[1]))
        self.canvas4.draw()

    def draw_plot_lster(self, x_min=0, x_max=sys.maxsize):
        self.ax5.clear()

        min_idx = np.searchsorted(self.seconds_window, x_min, side="left")
        max_idx = np.searchsorted(self.seconds_window, x_max, side="right")

        self.ax5.plot(self.seconds_window[min_idx:max_idx], self.calculated_lster[min_idx:max_idx])
        self.ax5.set_title('LSTER')
        self.ax5.set_xlim((self.ax1.get_xlim()[0], self.ax1.get_xlim()[1]))
        self.canvas5.draw()

    def reset_plots(self):
        self.onselect(0, len(self.sound))
        sd.stop()


app = QApplication([])
window = MainWindow()
window.setGeometry(100, 100, 1200, 800)
window.show()
app.exec_()

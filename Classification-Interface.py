import sys
from PyQt5.QtCore import Qt
import sounddevice
import soundfile
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Zahlen zu Klassen zuordnen und andersrum
label_to_int = {
    'Air Conditioner': 0, 
    'Car Horn': 1, 
    'Children Playing': 2, 
    'Dog Park': 3, 
    'Drilling': 4, 
    'Engine Idling': 5, 
    'Gun Shot': 6, 
    'Jackhammer': 7, 
    'Siren': 8, 
    'Street Music': 9
}
int_to_label = {v: k for k, v in label_to_int.items()}  # keys und values tauschen -> aus Original-Notebook entnommen

# Für die Klassifikation wird Mel-Spec gebraucht
# -> Umwandlung von wave file in mel-spec nötig
def wav_to_melspec(wav_path, duration=2.97, sr=22050, n_mels=128):
    y, _ = librosa.load(wav_path, sr=sr, duration=duration)
    length_soll = int(sr * duration)
    if len(y) < length_soll:
        y = np.pad(y, (0, length_soll - len(y)), mode='constant') # wenn array zu kurz ist, werden am Ende Nullen hinzugefügt
    else:
        y = y[:length_soll] # wenn array zu lang ist -> zugeschnitten (hinten)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return mel_spec

# Hauptfenster
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Classification Interface')
        self.setFixedSize(800, 800)
        
        self.model = load_model('best_model_trained_50epochs-v1.hdf5')  # Modell laden
        
        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        
        # Buttons
        hbox_buttons = QHBoxLayout()
        self.button_record = QPushButton('Aufnehmen')
        self.button_import = QPushButton('Importieren')
        self.button_play = QPushButton('Abspielen')
        self.button_classify = QPushButton('Klassifizieren')
        hbox_buttons.addWidget(self.button_record)
        hbox_buttons.addWidget(self.button_import)
        hbox_buttons.addWidget(self.button_play)
        hbox_buttons.addWidget(self.button_classify)
        vbox.addLayout(hbox_buttons)
        
        # Labels
        hbox_labels = QHBoxLayout()
        self.label_result = QLabel('Ergebnis: ')
        self.label_status = QLabel('Status: Bereit')
        hbox_labels.addWidget(self.label_result, stretch=1)
        hbox_labels.addWidget(self.label_status, stretch=1)
        vbox.addLayout(hbox_labels)
        
        # Plots
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        vbox.addWidget(self.canvas, stretch=1)
        
        # Verbindung zwischen Knopf und Funktion
        self.button_record.clicked.connect(self.record_function)
        self.button_import.clicked.connect(self.import_function)
        self.button_play.clicked.connect(self.play_function)
        self.button_classify.clicked.connect(self.classify_function)
        
        # Platzhalter für letzte Audio
        self.last_audio = None
        self.last_sr = None
        self.last_path = None
        
    # nimmt path zur wave file, erstellt mel_spec und macht predict und gibt label aus
    def predict_from_wav(self, path):
        ps = wav_to_melspec(path)
        X = ps.reshape(1, 128, 128, 1)
        y = int(self.model.predict(X).argmax(axis=1)[0]) # gibt label als Zahl aus -> argmax(axis=1): findet index des höchsten wertes in jeder Zeile
        y = int_to_label[y]
        return y
    
    def plots(self, y, sr):
        self.figure.clear()
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax1.plot(np.linspace(0, len(y)/sr, len(y)), y)
        ax1.set_ylabel('Amplitude')
        
        ax2 = self.figure.add_subplot(2, 1, 2)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
        # img = ax2.imshow(mel_spec, aspect='auto', origin='lower')
        img = librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
        ax2.set_xlabel('Zeit [s]')
        ax2.set_ylabel('Frequenz [Hz]')
        cbar = self.figure.colorbar(img, ax=ax2)
        cbar.ax.set_yticklabels([])
        cbar.ax.set_ylabel('')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def import_function(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Wähle eine WAV-Datei', filter='WAV Files (*.wav)')
        if not path:
            return
        y, sr = soundfile.read(path)    # Fürs Plotten zwischenspeichern
        self.last_audio = y.flatten()
        self.last_sr = sr
        self.last_path = path
        self.plots(y.flatten(), sr)
        self.label_status.setText('Status: Datei geladen')
        
    def record_function(self):
        self.label_status.setText('Status: Aufnahme...')
        duration = 2.97
        sr = 22050
        audio = sounddevice.rec(int(duration * sr), samplerate=sr, channels=1)
        sounddevice.wait()  # warten, bist Aufnahme abgeschlossen wurde
        y = audio.flatten()     # macht aus 2D array (n, 1) ein 1D array (n,)
        soundfile.write('recorded.wav', audio, sr)
        self.last_audio = y
        self.last_sr = sr
        self.last_path = 'recorded.wav'
        self.label_status.setText('Status: Aufnahme fertig')
        self.plots(y, sr)
        
    def classify_function(self):
        if self.last_path is None:
            self.label_status.setText('Status: Keine Audio!')
            return
        self.label_status.setText('Status: Klassifiziere...')
        label = self.predict_from_wav(self.last_path)
        self.label_result.setText(f'Ergebnis: {label}')
        self.label_status.setText('Status: Fertig')
        
    def play_function(self):
        if self.last_audio is None:
            self.label_status.setText('Status: Keine Audio')
            return
        sounddevice.play(self.last_audio, samplerate=self.last_sr)
        self.label_status.setText('Status: Läuft...')
        sounddevice.wait()
        self.label_status.setText('Status: Fertig')
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
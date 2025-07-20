import sys
from PyQt5.QtCore import Qt
import sounddevice
import soundfile
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog)

# Zahlen zu Klassen zuordnen und andersrum
label_to_int = {
    'air_conditioner': 0, 
    'car_horn': 1, 
    'children_playing': 2, 
    'dog_park': 3, 
    'drilling': 4, 
    'engine_idling': 5, 
    'gun_shot': 6, 
    'jackhammer': 7, 
    'siren': 8, 
    'street_music': 9
}
int_to_label = {v: k for k, v in label_to_int.items()}  # keys und values tauschen

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
    ps = librosa.power_to_db(mel_spec, ref=np.max)  # Umwandlung von Leistungsskala in dB Skala
    return ps

# Hauptfenster
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Classification Interface')
        self.setFixedSize(800, 400)
        
        self.model = load_model('best_model_trained.hdf5')  # Modell laden
        
        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        hbox = QHBoxLayout()
        self.button_record = QPushButton('Aufnehmen')
        self.button_import = QPushButton('Importieren')
        hbox.addWidget(self.button_record)
        hbox.addWidget(self.button_import)
        vbox.addLayout(hbox)
        
        self.label_result = QLabel('Ergebnis: ')
        vbox.addWidget(self.label_result)
        self.label_status = QLabel('Status: Bereit')
        vbox.addWidget(self.label_status)
        
        self.button_record.clicked.connect(self.record_function)
        self.button_import.clicked.connect(self.import_function)
        
    # nimmt path zur wave file, erstellt mel_spec und macht predict und gibt label aus
    def predict_from_wav(self, path):
        ps = wav_to_melspec(path)
        X = ps.reshape(1, 128, 128, 1)
        y = int(self.model.predict(X).argmax(axis=1)[0]) # gibt label als Zahl aus -> argmax(axis=1): findet index des höchsten wertes in jeder Zeile
        y = int_to_label[y]
        return y
    
    def import_function(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Wähle eine WAV-Datei', filter='WAV Files (*.wav)')
        if not path:
            return
        self.label_status.setText('Status: Klassifiziere...')
        label = self.predict_from_wav(path)
        self.label_result.setText(f'Ergebnis: {label}')
        self.label_status.setText('Status: Fertig')
        
    def record_function(self):
        self.label_status.setText('Status: Aufnahme...')
        duration = 2.97
        sr = 22050
        audio = sounddevice.rec(int(duration * sr), samplerate=sr, channels=1)
        sounddevice.wait()  # warten, bist Aufnahme abgeschlossen wurde
        temp = 'recorded.wav'
        soundfile.write(temp, audio, sr)
        self.label_status.setText('Status: Klassifiziere Aufnahme...')
        label = self.predict_from_wav(temp)
        self.label_result.setText(f'Ergebnis: {label}')
        self.label_status.setText('Status: Fertig')
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
#!/usr/bin/env bash         -> Sagt dem Betriebssystem, welcher Interpreter das Skript ausführen soll
set -euo pipefail       # Verhindert, dass ein späterer Teil des Skripts weiterläuft, obwohl zuvor etwas schiefgegangen ist

# Python-Package installieren
echo "Stelle sicher, dass soundata installiert ist"
python3 -m pip install --upgrade soundata

# Zielordner anlegen
DATA_HOME="data"
mkdir -p "${DATA_HOME}"

# Dataset herunterladen mit Python-API
echo "Lade UrbanSound8K per Python-API herunter"
python3 - <<EOF
import os
import soundata

# Lege data-home fest
data_home = os.path.abspath("data")
# Initialisiere den Loader für UrbanSound8K
dataset = soundata.initialize("urbansound8k", data_home=data_home)
# Lade (und entpacke) die Audiodaten
dataset.download()
print(f"UrbanSound8K wurde in {data_home}/urban_sound_8k abgelegt.")
EOF

echo "Entferne das Archiv, um Speicher zu sparen"
rm -f "${DATA_HOME}/urban_sound_8k/UrbanSound8K.tar.gz"

echo "Fertig!"

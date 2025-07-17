#!/usr/bin/env bash
set -euo pipefail

# 1) Python-Package installieren (im aktiven Environment)
echo "Stelle sicher, dass soundata installiert ist…"
python3 -m pip install --upgrade soundata

# 2) Zielordner anlegen
DATA_HOME="data"
mkdir -p "${DATA_HOME}"

# 3) Dataset herunterladen mit Python-API
echo "Lade UrbanSound8K per Python-API herunter…"
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

echo "Fertig!"

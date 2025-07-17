#!/usr/bin/env bash
set -euo pipefail

# 1) Virtual Env aktivieren oder sicherstellen, dass soundata installiert ist
if ! command -v soundata &> /dev/null; then
  echo "soundata nicht gefunden. Installiere via pip..."
  pip install soundata
fi

# 2) Ziel-Ordner
DATA_DIR="data/UrbanSound8K"
mkdir -p "${DATA_DIR}"

# 3) Dataset herunterladen
echo "Lade UrbanSound8K mit soundata herunter..."
python3 - <<EOF
import soundata
# download() holt den Datensatz und legt ihn unter DATA_HOME ab, Standard: ~/soundata
# Wir setzen DATA_HOME auf unser Projekt-Verzeichnis
import os
os.environ['SOUNDATA_HOME'] = os.path.abspath('data')
soundata.download('urban_sound_8k')
EOF

echo "Fertig! Die Daten stehen jetzt in ${DATA_DIR}"

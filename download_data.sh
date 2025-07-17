#!/usr/bin/env bash
set -euo pipefail

# 1) Stelle sicher, dass das soundata-CLI verfügbar ist
if ! command -v soundata &> /dev/null; then
  echo "soundata CLI nicht gefunden. Installiere via pip..."
  pip install soundata
fi

# 2) Zielordner anlegen
DATA_DIR="data"
mkdir -p "${DATA_DIR}"

# 3) Dataset per soundata-CLI herunterladen
echo "Lade UrbanSound8K mit soundata CLI herunter..."
soundata download urban_sound_8k --data-home "${DATA_DIR}"

echo "Fertig! Die Daten stehen jetzt in ${DATA_DIR}/urban_sound_8k"

# Urban Sound Classification

In diesem Projekt wird ein Convolutional Neural Network (CNN) auf Basis von [Min Yang](https://www.kaggle.com/code/mychen76/automatic-urban-sound-classification-with-cnn) trainiert. 

## Repository klonen in bash

git clone https://github.com/pascalsto/Urban-Sound-Classification

## Erstellen eines Environments mit allen benötigten Paketen (conda benötigt)

conda env create -f environment.yml
conda activate urban-sound-py310

## UrbanSound8K-Datensatz mit bash runterladen

chmod +x download_data.sh<br>
bash download_data.sh

## Datensätze für Datenerweiterung

[HornBase - A Car Horns Dataset](https://data.mendeley.com/datasets/y5stjsnp8s/2)<br>
[Gunshot audio dataset](https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset)

## Allgemeine Infos

- Kaggle-Classification.ipynb ist das Original-Notebook von Kaggle
- Classification.ipynb ist mein bearbeitetes Notebook
- Classification-Interface.py ist das Skript, womit das Modell mit eigenen Tonaufnahmen getestet werden kann
- für das Interface wurde das Modell best_model_trained_50epochs-v1.hdf5 verwendet
- die requirements.txt Datei enthält wichtigsten Pakete, welche für das Laufen des Codes benötigt werden
- recorded.wav ist die selbstaufgenommene Tonspur aus dem Klassifikations-Interface
- data, .DS_Store und *.p in .gitignore geschrieben, damit die Daten (über 8 Gb) nicht ins Repository hochgeladen werden
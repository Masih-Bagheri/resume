# Alzheimer-MRT-Klassifikation — Deutsch

[Projektübersicht](../README.md) · [English](../eng/README.md) · [فارسی](../per/README.md)

Dieses Hochschulprojekt klassifiziert MRT-Bilder in vier Datensatzklassen mit KNN. Die Bilder werden in Graustufen umgewandelt und auf 64 × 64 Pixel skaliert. StandardScaler und PCA verarbeiten die Merkmale. Eine stratifizierte Kreuzvalidierung wählt die KNN-Parameter aus.

## Installation und Training

Öffnen Sie ein Terminal im Ordner Alzheimer, nicht im Ordner ger:

```bash
python -m pip install -r requirements.txt
python src/train.py
```

Laden Sie den Datensatz gemäß der [Datensatzanleitung](../data/DATASETS.md) herunter. Geben Sie den Pfad zum Ordner ein, der direkt Train und Val enthält. Beide Ordner benötigen die vier Klassenunterordner.

Das Training speichert models/knn_model.joblib sowie den Bericht und die Konfusionsmatrix im Ordner results. Die Konsolenausgabe des Trainings ist auf Englisch.

## Anwendung starten

```bash
python src/app.py
```

Wählen Sie Deutsch und klicken Sie auf MRT-Bild auswählen. Die vorhergesagte Klasse und ihre ursprüngliche Kennung werden angezeigt. Ein Sprachwechsel übersetzt auch das aktuelle Ergebnis; erneutes Training ist nicht erforderlich. Das optionale Hintergrundbild liegt unter assets/Brain.png.

Wenn das Modell fehlt, erscheint beim Anklicken der Schaltfläche ein Hinweis. Das Fenster bleibt geöffnet. Unlesbare Bilder erzeugen eine Fehlermeldung. Technische Fehlermeldungen von Bibliotheken können auf Englisch bleiben.

## Beispielergebnisse

430 Trainingsbilder und 364 Validierungsbilder; Genauigkeit 54,12 %; Fehlerrate 45,88 %. Diese Werte stammen aus einer kleinen Entwicklungsstichprobe, nicht aus dem vollständigen Datensatz. [Klassifikationsbericht](../results/evaluation.txt).

## Klassen

| Class identifier | English | فارسی | Deutsch |
| --- | --- | --- | --- |
| MildDemented | Mild dementia | زوال عقل خفیف | Leichte Demenz |
| ModerateDemented | Moderate dementia | زوال عقل متوسط | Mittelschwere Demenz |
| NonDemented | Non-demented | بدون زوال عقل | Keine Demenz |
| VeryMildDemented | Very mild dementia | زوال عقل بسیار خفیف | Sehr leichte Demenz |

## Grenzen

Das Projekt dient dem Lernen und ist nicht klinisch validiert. Die Anwendung prüft nicht, ob die Eingabe ein MRT-Bild ist. Eine Trennung nach Patienten wurde nicht bestätigt. Bilder desselben Patienten und augmentierte Varianten müssen innerhalb derselben Aufteilung bleiben. Die gemessene Genauigkeit belegt keine medizinische Zuverlässigkeit.

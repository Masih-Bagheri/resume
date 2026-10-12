# Alzheimer MRI Classification — English

[Project overview](../README.md) · [فارسی](../per/README.md) · [Deutsch](../ger/README.md)

This university project trains KNN to classify four MRI dataset categories. It uses grayscale 64 × 64 images, StandardScaler, PCA, and cross-validation for KNN parameter selection.

## Install and train

Open a terminal in the Alzheimer folder, not eng:

```bash
python -m pip install -r requirements.txt
python src/train.py
```

Get the dataset using [the dataset guide](../data/DATASETS.md). Enter the path to its root containing Train and Val, each containing the four class subfolders listed below. Training saves models/knn_model.joblib, results/evaluation.txt, and results/confusion_matrix.png.

## Run

```bash
python src/app.py
```

Choose English in the language selector and click Select an MRI image. The predicted class and original class identifier appear. Changing language also translates the current prediction without retraining. The optional background lives at assets/Brain.png.

A missing model is reported when you click the image button; the window stays open. Unreadable images produce an error dialog. Detailed library errors may be in English.

## Sample evaluation

430 training images; 364 validation images; accuracy 54.12%; error rate 45.88%. These are small-sample development results. See [the report](../results/evaluation.txt).

## Classes

| Class identifier | English | فارسی | Deutsch |
| --- | --- | --- | --- |
| MildDemented | Mild dementia | زوال عقل خفیف | Leichte Demenz |
| ModerateDemented | Moderate dementia | زوال عقل متوسط | Mittelschwere Demenz |
| NonDemented | Non-demented | بدون زوال عقل | Keine Demenz |
| VeryMildDemented | Very mild dementia | زوال عقل بسیار خفیف | Sehr leichte Demenz |

## Limitations

Educational use only; no clinical validation. The application does not verify whether an input is an MRI. Patient-level separation has not been established. Do not mix patients or related augmented images across evaluation splits. Accuracy is a sample evaluation score, not medical reliability.

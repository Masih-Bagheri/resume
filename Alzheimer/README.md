# Alzheimer MRI Classification with KNN

University project with a shared Python source and an English / فارسی / Deutsch desktop interface.

## Choose a guide

- [English](eng/README.md)
- [فارسی](per/README.md)
- [Deutsch](ger/README.md)

The language selector changes the window title, buttons, dialogs, result labels, and educational notice. Original dataset class identifiers stay unchanged. Third-party exception details may remain in English. Training console output is in English.

## Quick start

Run these commands from this Alzheimer folder:

```bash
python -m pip install -r requirements.txt
python src/train.py
python src/app.py
```

Training asks for the dataset root containing Train and Val. See [dataset instructions](data/DATASETS.md). The repository includes neither dataset images nor a pretrained model. The interface opens without a model; selecting an image prompts you to train first if the model is missing. Existing compatible models from the previous version can be reused without retraining.

## Method and results

64 × 64 grayscale images → StandardScaler → PCA retaining 95% explained variance → KNN. Three-fold stratified cross-validation selects KNN settings using macro F1, fitting the full pipeline inside each fold.

| Sample result | Value |
| --- | --- |
| Training images | 430 |
| Validation images | 364 |
| Validation accuracy | 54.12% |
| Validation error rate | 45.88% |
| Training CV macro F1 | Approximately 0.508 |
| Selected KNN settings | n_neighbors=1, weights=uniform, p=1 |

These metrics belong to the small development sample, not the full dataset. See [classification report](results/evaluation.txt) and [confusion matrix](results/confusion_matrix.png).

## Files

| Path | Purpose |
| --- | --- |
| src/train.py | Training, evaluation, and saving the complete pipeline |
| src/app.py | Shared desktop application with three languages |
| eng/, per/, ger/ | Language-specific guides |
| assets/Brain.png | Optional resizable background |
| data/DATASETS.md | Dataset source and required folder layout |
| models/ | Local generated model; excluded from Git |
| results/ | Sample evaluation results |
| BUILD_WINDOWS.md | Optional executable build instructions |

## Verification and limitations

The preceding source and executable were manually run on the author's Windows machine. The new multilingual interface has undergone automated checks without a real display; visual checks on Windows are still needed. No test on a separate machine without Python has been confirmed. Dependencies are not fully version-pinned.

This educational project is not clinically validated. It does not reject non-MRI images. Subject-level separation and sample provenance have not been verified. Keep patients and augmented versions of an original image within one split. Development validation scores are not an untouched final test result. Error rate is an aggregate metric, not confidence in a particular prediction.

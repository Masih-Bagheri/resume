# Dataset guide

## Source used for this project

[Augmented Alzheimer MRI Dataset V2 — uraninjo on Kaggle](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset-v2)

The dataset is hosted on Kaggle, not GitHub. Follow the source's download instructions and terms. The full dataset is not included in this repository; reported development results used only a small local sample.

## Required local structure

Choose a dataset root such as data/AlzheimerDataset. It must contain:

| Split | Required class subfolders |
| --- | --- |
| Train | MildDemented, ModerateDemented, NonDemented, VeryMildDemented |
| Val | MildDemented, ModerateDemented, NonDemented, VeryMildDemented |

Put JPG, JPEG, or PNG images directly inside each class subfolder. Split and class folder matching is case-insensitive in the current loader. Images nested more deeply are not loaded.

Run python src/train.py and enter the root folder path, not a class folder or Train itself. Example on Windows:

```text
D:\resume\Alzheimer\data\AlzheimerDataset
```

## Other datasets

Other datasets require compatible labels and the same directory layout. A dataset with different categories requires code changes and retraining. Add further verified sources here as the project develops.

Keep evaluation images separate from training images. If subject identifiers are available, split by subject. For augmented images, split by the original source before augmentation. Do not merge Train and Val to improve the reported score.

## Git tracking

Keep these lines in the root .gitignore so the guide is tracked while local dataset images are excluded:

```gitignore
data/*
!data/.gitkeep
!data/DATASETS.md
```

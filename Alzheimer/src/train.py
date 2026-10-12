from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_SIZE = (64, 64)

CLASSES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]


def find_folder(parent, name):
    for folder in parent.iterdir():
        if folder.is_dir() and folder.name.lower() == name.lower():
            return folder

    raise FileNotFoundError(f"Folder not found: {parent / name}")


def read_image(path):
    file_data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(file_data, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Cannot read image")

    img = cv2.resize(img, IMAGE_SIZE)
    return img.flatten()


def load_dataset(folder):
    data = []
    targets = []

    for class_number, class_name in enumerate(CLASSES):
        class_folder = find_folder(folder, class_name)
        count = 0

        for path in sorted(class_folder.iterdir()):
            if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                data.append(read_image(path))
                targets.append(class_number)
                count += 1
            except Exception as error:
                print(f"Skipped: {path.name} | {error}")

        print(f"{folder.name} / {class_name}: {count}")

        if count == 0:
            raise ValueError(f"No readable images in {class_folder}")

    return np.array(data), np.array(targets)


def main():
    entered_path = input("Enter dataset folder path: ").strip().strip('"')
    dataset_path = Path(entered_path).expanduser()

    if not dataset_path.is_dir():
        raise FileNotFoundError("Dataset folder does not exist")

    train_folder = find_folder(dataset_path, "train")
    val_folder = find_folder(dataset_path, "val")

    X_train, y_train = load_dataset(train_folder)
    X_val, y_val = load_dataset(val_folder)

    print("\nTraining KNN...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("knn", KNeighborsClassifier()),
    ])

    parameters = {
        "knn__n_neighbors": [1, 3, 5, 7, 9],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        pipeline,
        parameters,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_

    print("Best parameters:", search.best_params_)
    print(f"Training CV macro F1: {search.best_score_:.3f}")

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    error_rate = 1 - accuracy

    report = classification_report(
        y_val,
        predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
    )

    summary = (
        f"Training images: {len(X_train)}\n"
        f"Validation images: {len(X_val)}\n"
        f"Accuracy: {accuracy * 100:.2f}%\n"
        f"Error rate: {error_rate * 100:.2f}%\n"
    )

    print("\n" + summary)
    print(report)

    models_dir = PROJECT_DIR / "models"
    results_dir = PROJECT_DIR / "results"

    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    model_path = models_dir / "knn_model.joblib"

    joblib.dump(
        {
            "model": model,
            "classes": CLASSES,
            "image_size": IMAGE_SIZE,
        },
        model_path,
    )

    (results_dir / "evaluation.txt").write_text(
        summary + "\n" + report,
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ConfusionMatrixDisplay.from_predictions(
        y_val,
        predictions,
        labels=list(range(len(CLASSES))),
        display_labels=CLASSES,
        cmap="Blues",
        xticks_rotation=30,
        ax=ax,
        colorbar=False,
    )

    ax.set_title("KNN - Validation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"Model saved: {model_path}")
    print(f"Results saved: {results_dir}")


if __name__ == "__main__":
    main()
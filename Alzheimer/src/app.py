from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk


if getattr(sys, "frozen", False):
    PROJECT_DIR = Path(sys.executable).resolve().parent
else:
    PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "knn_model.joblib"
BACKGROUND_PATH = PROJECT_DIR / "assets" / "Brain.png"

LANGUAGES = {"English": "eng", "فارسی": "per", "Deutsch": "ger"}
TRANSLATIONS = {
    "eng": {
        "title": "Alzheimer MRI Classification — KNN",
        "heading": "MRI image classification with KNN",
        "language": "Language",
        "select": "Select an MRI image",
        "initial": "Select an image to see its predicted class",
        "result": "Predicted class: {label}",
        "notice": "Educational project — not clinically validated",
        "image_files": "Image files",
        "error": "Error",
        "unreadable": "The selected image cannot be read.",
        "missing_model": "Model not found. Run src/train.py first.\n\nExpected path:\n{path}",
        "load_error": "Could not load the model:\n{error}",
        "predict_error": "Could not predict this image:\n{error}",
        "background_error": "Could not read the optional background:\n{error}",
        "classes": {
            "MildDemented": "Mild dementia",
            "ModerateDemented": "Moderate dementia",
            "NonDemented": "Non-demented",
            "VeryMildDemented": "Very mild dementia",
        },
    },
    "per": {
        "title": "طبقه‌بندی تصاویر MRI آلزایمر — KNN",
        "heading": "طبقه‌بندی تصویر MRI با KNN",
        "language": "زبان",
        "select": "انتخاب تصویر MRI",
        "initial": "برای نمایش کلاس پیش‌بینی‌شده، یک تصویر انتخاب کن",
        "result": "کلاس پیش‌بینی‌شده: {label}",
        "notice": "پروژهٔ آموزشی — اعتبارسنجی بالینی نشده",
        "image_files": "فایل‌های تصویری",
        "error": "خطا",
        "unreadable": "تصویر انتخاب‌شده قابل خواندن نیست.",
        "missing_model": "مدل پیدا نشد. ابتدا src/train.py را اجرا کن.\n\nمسیر مورد انتظار:\n{path}",
        "load_error": "مدل بارگذاری نشد:\n{error}",
        "predict_error": "پیش‌بینی تصویر انجام نشد:\n{error}",
        "background_error": "تصویر پس‌زمینه خوانده نشد:\n{error}",
        "classes": {
            "MildDemented": "زوال عقل خفیف",
            "ModerateDemented": "زوال عقل متوسط",
            "NonDemented": "بدون زوال عقل",
            "VeryMildDemented": "زوال عقل بسیار خفیف",
        },
    },
    "ger": {
        "title": "Alzheimer-MRT-Klassifikation — KNN",
        "heading": "Klassifikation von MRT-Bildern mit KNN",
        "language": "Sprache",
        "select": "MRT-Bild auswählen",
        "initial": "Wählen Sie ein Bild für die Klassifikation aus",
        "result": "Vorhergesagte Klasse: {label}",
        "notice": "Lernprojekt — nicht klinisch validiert",
        "image_files": "Bilddateien",
        "error": "Fehler",
        "unreadable": "Das ausgewählte Bild kann nicht gelesen werden.",
        "missing_model": "Modell nicht gefunden. Führen Sie zuerst src/train.py aus.\n\nErwarteter Pfad:\n{path}",
        "load_error": "Das Modell konnte nicht geladen werden:\n{error}",
        "predict_error": "Die Klassifikation ist fehlgeschlagen:\n{error}",
        "background_error": "Das Hintergrundbild konnte nicht gelesen werden:\n{error}",
        "classes": {
            "MildDemented": "Leichte Demenz",
            "ModerateDemented": "Mittelschwere Demenz",
            "NonDemented": "Keine Demenz",
            "VeryMildDemented": "Sehr leichte Demenz",
        },
    },
}


def main():
    root = tk.Tk()
    root.geometry("1000x600")
    root.minsize(760, 440)
    root.configure(bg="#122B40")

    language = tk.StringVar(value="English")
    model_bundle = None
    predicted_label = None

    def texts():
        return TRANSLATIONS[LANGUAGES[language.get()]]

    background_label = tk.Label(root, bg="#122B40", borderwidth=0)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    original_background = None

    if BACKGROUND_PATH.is_file():
        try:
            with Image.open(BACKGROUND_PATH) as img:
                original_background = img.convert("RGB")
        except Exception as error:
            messagebox.showwarning(
                texts()["error"],
                texts()["background_error"].format(error=error),
                parent=root,
            )

    def resize_background(event=None):
        if original_background is None:
            return
        if event is not None and event.widget is not root:
            return
        size = (max(root.winfo_width(), 1), max(root.winfo_height(), 1))
        resized = original_background.resize(size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        background_label.configure(image=photo)
        background_label.image = photo

    panel = tk.Frame(root, bg="white", padx=25, pady=25)
    panel.place(relx=0.5, rely=0.5, anchor="center")

    language_row = tk.Frame(panel, bg="white")
    language_row.pack(pady=(0, 15))
    language_label = tk.Label(language_row, bg="white", font=("Arial", 11))
    language_label.pack(side="left", padx=10)
    language_picker = ttk.Combobox(
        language_row,
        textvariable=language,
        values=list(LANGUAGES),
        state="readonly",
        width=12,
    )
    language_picker.pack(side="left")

    heading = tk.Label(panel, font=("Arial", 18, "bold"), bg="white")
    heading.pack(pady=(0, 20))
    result_label = tk.Label(
        panel, font=("Arial", 14), bg="white", fg="#122B40", wraplength=620
    )
    filename_label = tk.Label(
        panel, text="", font=("Arial", 10), bg="white", wraplength=620
    )
    class_code_label = tk.Label(
        panel, text="", font=("Arial", 10), bg="white", fg="#666666"
    )

    def open_file():
        nonlocal model_bundle, predicted_label
        t = texts()

        if model_bundle is None:
            if not MODEL_PATH.is_file():
                messagebox.showerror(
                    t["error"], t["missing_model"].format(path=MODEL_PATH), parent=root
                )
                return
            try:
                bundle = joblib.load(MODEL_PATH)
                for key in ("model", "classes", "image_size"):
                    if key not in bundle:
                        raise ValueError(f"Missing model field: {key}")
                model_bundle = bundle
            except Exception as error:
                messagebox.showerror(
                    t["error"], t["load_error"].format(error=error), parent=root
                )
                return

        file_path = filedialog.askopenfilename(
            parent=root,
            title=t["select"],
            filetypes=[(t["image_files"], "*.jpg *.jpeg *.png")],
        )
        if not file_path:
            return

        try:
            path = Path(file_path)
            file_data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            img = cv2.imdecode(file_data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(t["unreadable"])
            img = cv2.resize(img, model_bundle["image_size"])
            features = img.flatten().reshape(1, -1)
            class_index = int(model_bundle["model"].predict(features)[0])
            predicted_label = model_bundle["classes"][class_index]
            filename_label.configure(text=path.name)
            update_language()
        except Exception as error:
            messagebox.showerror(
                t["error"], t["predict_error"].format(error=error), parent=root
            )

    select_button = tk.Button(
        panel, command=open_file, font=("Arial", 13, "bold"),
        bg="#DDEEEF", padx=20, pady=8,
    )
    select_button.pack(pady=10)
    result_label.pack(pady=15)
    class_code_label.pack()
    filename_label.pack(pady=(5, 0))
    notice_label = tk.Label(
        panel, font=("Arial", 10), bg="white", fg="#666666", wraplength=620
    )
    notice_label.pack(pady=(20, 0))

    def update_language(event=None):
        t = texts()
        root.title(t["title"])
        language_label.configure(text=t["language"])
        heading.configure(text=t["heading"])
        select_button.configure(text=t["select"])
        notice_label.configure(text=t["notice"])
        if predicted_label is None:
            result_label.configure(text=t["initial"])
        else:
            translated = t["classes"].get(predicted_label, predicted_label)
            result_label.configure(text=t["result"].format(label=translated))
            class_code_label.configure(text=predicted_label)

    language_picker.bind("<<ComboboxSelected>>", update_language)
    update_language()
    root.bind("<Configure>", resize_background)
    root.after(100, resize_background)
    root.mainloop()


if __name__ == "__main__":
    main()

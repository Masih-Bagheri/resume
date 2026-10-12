# Optional Windows build

Running from source is the primary distribution method. The previous interface was successfully packaged on the author's Windows machine. Rebuild and test this new multilingual version before sharing an executable. Build in the same environment used for model training.

From the Alzheimer folder:

```bat
python -m pip install pyinstaller
python -m PyInstaller --clean --onefile --windowed --name AlzheimerKNN --collect-submodules sklearn src/app.py
mkdir dist\models
mkdir dist\assets
copy models\knn_model.joblib dist\models\knn_model.joblib
copy assets\Brain.png dist\assets\Brain.png
```

The model is required; the background is optional. app.py already reads resources from folders next to the executable. Distribute the entire dist folder, without dataset images. Its size can be approximately 250 MB due to bundled dependencies.

Test language switching, prediction, resizing, and unreadable files. Test again after moving the whole folder. A separate Windows machine without Python is needed to confirm portability. If the executable will not start, replace --windowed with --console to see diagnostic output.

Do not upload build or dist into the source repository. Optionally ZIP dist for a GitHub release.

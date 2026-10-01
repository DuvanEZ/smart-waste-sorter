# Smart Waste Sorter – User Guide

## 1. What the application does

Smart Waste Sorter looks at a photo of **one waste item** and predicts its material:

| Class | Examples | Disposal (NZ standard kerbside rules) |
|---|---|---|
| cardboard | boxes, packaging (drink cartons look similar but are **not** accepted at kerbside) | Recycling bin (paper & cardboard) |
| glass | bottles, jars | Glass crate / glass recycling |
| metal | aluminium and steel cans, tins | Recycling bin (cans) |
| paper | newspaper, office paper, magazines | Recycling bin (paper & cardboard) |
| plastic | bottles and containers | Recycling bin – plastics 1, 2 and 5 only |
| trash | non-recyclable waste | General rubbish (landfill) |

For every image it reports the predicted class, how confident the model is, the other likely
classes and where the item should go.

## 2. Installation and start-up

**Requirements:** Windows, macOS or Linux; **Python 3.10 – 3.13** (<https://www.python.org/downloads/>);
internet access the first time only. No GPU is needed.

| Step | Windows | macOS / Linux |
|---|---|---|
| 1. Unzip the project folder | Right-click → *Extract all* | double-click the zip |
| 2. Start the app | double-click **`run_app.bat`** | in a terminal inside the folder: `./run_app.sh` |
| 3. Use the app | the browser opens **http://localhost:8501** | same |
| 4. Stop the app | close the black console window | press `Ctrl + C` in the terminal |

The first start creates a private Python environment (`.venv`) and installs the libraries from
`requirements.txt` (1–3 minutes). Later starts take a few seconds.

**Manual start (any system):**
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
If the browser does not open automatically, copy the *Local URL* printed in the terminal
(normally http://localhost:8501) into the browser.

**Command-line alternative (no browser):** `python app/cli.py` shows a menu (classify one image,
classify a folder, change the threshold, help, quit). `python app/cli.py photo.jpg` or
`python app/cli.py my_folder --csv results.csv` run without the menu.

## 3. Using the web application

### Sidebar
* **Confidence threshold** (default 60 %): predictions with a lower probability are labelled *UNCERTAIN*.
* **Show the probability of every class**: show all six classes instead of the top three.
* **Model**: test accuracy and macro-F1 of the model.

### Tab 1 – Classify an item
Choose how to provide the picture:
1. **Upload a photo** – click *Browse files* or drag an image into the box.
2. **Take a photo with the webcam** – allow camera access in the browser, then click *Take photo*.
3. **Use a sample image** – choose one of the test images shipped in `sample_images/`.

The result card shows the photo, the predicted class (e.g. **GLASS**), a confidence bar, a green
*Confident prediction* or yellow *UNCERTAIN* message, the correct bin with recycling tips, and a table
with the probability of each class. When the prediction is *UNCERTAIN*, the card names the second most
likely class and phrases the advice conditionally ("Where does it go if it is …?") – check the item yourself
or take a clearer photo.

### Tab 2 – Batch classification
* **Upload several files** – select many images, then click *Classify uploaded images*; or
* **Classify a folder on this computer** – type or paste a folder path, e.g. `C:\Users\me\Pictures\waste`
  or `/Users/me/waste`, then click *Classify folder* (all supported images directly in the folder, max. 500).

The tab shows counts (files, confident, uncertain, rejected), a results table (one row per file), a red
explanation for every rejected file and a blue note for every flagged file, a **Download results as CSV**
button and a bar chart of the number of items per predicted material.

### Tab 3 – Model performance
Test-set accuracy, macro-F1, top-2 accuracy, ROC-AUC, per-class precision/recall/F1 and the confusion matrix.

### Tab 4 – Help
A short version of this guide.

## 4. Input format and validation

| Accepted | Rejected (with a message) | Accepted with a warning |
|---|---|---|
| `.jpg .jpeg .png .bmp .webp .tif .tiff` | other file types (e.g. `.txt`, `.pdf`, `.heic`) | grey-scale images |
| up to 20 MB | empty or corrupted files | transparent images (filled with white) |
| at least 32 × 32 px | images smaller than 32 px per side | low resolution (< 128 px) |
| any orientation (EXIF rotation is applied) | extreme shapes (aspect ratio > 6 : 1) | animated images (first frame used), very elongated images |
| | non-existent / empty folders | |

For the best results photograph **one item**, well lit, filling most of the frame, on a plain background.

## 5. Meaning of the outputs

| Output | Meaning |
|---|---|
| **Prediction** | the most likely material (one of the six classes) |
| **Confidence** | calibrated probability (0–1 or %) that the prediction is correct |
| **Status** | `OK` = confidence ≥ threshold · `UNCERTAIN` = below threshold, check manually · `REJECTED` = invalid input |
| **Second choice** | the next most likely class and its probability |
| **Disposal** | the bin to use under New Zealand's standard kerbside rules (check your council) |
| **Message** | the reason for a rejection or a warning about the input |

CSV columns (web app): `File, Prediction, Confidence, Second choice, Status, Disposal, Message`.
CSV columns (command line): `file, prediction, confidence, status, second_choice, disposal, message`.

## 6. Troubleshooting

| Problem | Solution |
|---|---|
| "Python 3.10 or newer is required" | install Python from python.org and tick *Add python.exe to PATH* (Windows) |
| installation fails | check the internet connection and run the launcher again (it resumes) |
| browser does not open | open http://localhost:8501 manually (or the URL printed in the terminal) |
| "port 8501 is in use" | Streamlit picks the next port (8502, …) – use the URL printed in the terminal |
| "The model could not be loaded" | keep the folders `app/` and `model/` together; do not rename the model files |
| webcam not available | allow camera access in the browser, or upload a photo instead |

## 7. Libraries

streamlit (web interface), onnxruntime (runs the neural network), numpy (arrays), pillow (images),
pandas (tables and CSV). Training additionally used PyTorch, torchvision, timm, scikit-learn,
scikit-image, OpenCV, matplotlib, imagehash, onnx and kagglehub.

// Section 3 (software), conclusion, references and appendices
const { Paragraph, TextRun, PageBreak } = require("docx");
const L = require("./lib");
const { p, h1, h2, h3, bullets, numbered, figure, table, code, figRef, tabRef, reference } = L;

function section31(R) {
  const out = [];
  out.push(h1("3. Software Implementation"));
  out.push(h2("3.1 Complete Source Code for the Application"));
  out.push(p("The trained model is embedded in **Smart Waste Sorter**, an interactive application written in Python. The main interface is a local web application built with **Streamlit**: it runs on the user's own computer (nothing is uploaded to the internet) and opens in the browser. A **command-line interface** with a text menu offers the same functions for computers without a browser. Both interfaces share the same prediction and validation code. The complete source code is in the zip archive of Appendix A and in the private GitHub repository `DuvanEZ/smart-waste-sorter`."));
  out.push(h3("Architecture and design principles"));
  out.push(...table("modules", "Modules of the Application (folder app/)",
    ["Module", "Responsibility"], [
      ["`streamlit_app.py`", "User interface: four tabs (classify one item, batch classification, model performance, help), sidebar settings, result cards, CSV export"],
      ["`cli.py`", "Command-line interface: interactive menu or one-off command with arguments; prints tables and saves CSV"],
      ["`predictor.py`", "`WasteClassifier` class: loads the ONNX model and its metadata, pre-processes images exactly as in training, runs inference, applies temperature scaling and returns `Prediction` objects (class, confidence, ranking, uncertainty flag)"],
      ["`validation.py`", "Validates every input (file type, size, integrity, dimensions, aspect ratio, colour mode, folder contents) and returns a clear accept/reject result with warnings"],
      ["`recycling_guide.py`", "Domain knowledge: disposal instructions for each class under New Zealand's standard kerbside rules"],
      ["`model/`", "`waste_classifier.onnx` (the trained network) and `model_info.json` (class names, normalisation, temperature, test metrics)"],
    ], [26, 74], null, { size: 17 }));
  out.push(p("The code follows established software-design principles:"));
  out.push(...bullets([
    "**Separation of concerns / single responsibility** – user interface, inference, validation and domain rules are separate modules, so each can be changed or tested independently (e.g. the CLI reuses the predictor and the validator unchanged).",
    "**Configuration instead of hard-coding** – class names, input size, normalisation constants, calibration temperature and default threshold are read from `model_info.json`, written automatically when the model is exported; retraining the model requires no code change.",
    "**Minimal, portable dependencies** – thanks to the ONNX export the application needs only Streamlit, ONNX Runtime, NumPy, Pillow and pandas (no PyTorch, no GPU); one-click launchers (`run_app.bat`, `run_app.sh`) create an isolated virtual environment and install them.",
    "**Defensive programming** – every file is validated before inference, errors are reported as readable messages instead of crashes, the model is loaded once and cached (`st.cache_resource`), and a missing model produces instructions rather than a stack trace.",
    "**Documentation and readability** – every module, class and function has a docstring; comments explain *why* (e.g. why the resize rule matches torchvision).",
    "**Testing** – 14 automated `pytest` tests cover validation rules, the predictor and the equivalence of the application's pre-processing with the training transform; all pass.",
  ]));
  out.push(h3("Interactivity and data entry"));
  out.push(p("The application prompts the user for data in ways that suit an image problem, and reports a prediction for each input:"));
  out.push(...bullets([
    "**One item** – upload a photo (file picker or drag-and-drop), take a photo with the webcam, or choose a sample test image;",
    "**Many items** – upload several photos at once or type the path of a folder (up to 500 images);",
    "**Settings** – a slider sets the confidence threshold below which a prediction is flagged *UNCERTAIN*; a checkbox shows all class probabilities;",
    "**Outputs** – predicted material, calibrated confidence, full probability ranking, uncertainty warning with the second most likely class, disposal instructions and recycling tips; batch results as a table with an explanation for every rejected or flagged file, a bar chart and a downloadable CSV file.",
  ]));
  out.push(h3("Input validation"));
  out.push(...table("validation", "Validation Rules Applied to Every Input",
    ["Check", "Rule", "Result if violated"], [
      ["File type", "extension must be jpg, jpeg, png, bmp, webp, tif or tiff", "Rejected with the list of allowed types"],
      ["Empty / too large", "1 byte to 20 MB", "Rejected"],
      ["Integrity", "the file must decode as an image (corrupted or fake files fail `verify()`)", "Rejected: not a valid image"],
      ["Decompression bomb", "at most 60 million pixels", "Rejected"],
      ["Minimum size", "at least 32 × 32 pixels", "Rejected: too small"],
      ["Aspect ratio", "at most 6 : 1", "Rejected: photograph a single item"],
      ["Colour mode", "grey-scale, palette or transparent images are converted to RGB", "Accepted with a warning"],
      ["Low resolution", "shorter side < 128 px", "Accepted with a warning"],
      ["Orientation", "EXIF rotation from phone cameras is applied", "Corrected automatically"],
      ["Folder path", "must exist, be a folder and contain supported images", "Error message"],
      ["Threshold (CLI)", "number between 30 and 95 %", "Asked again"],
    ], [20, 50, 30], null, { size: 17 }));
  out.push(h3("Key code"));
  out.push(p("The pre-processing in `predictor.py` reproduces the training transform exactly (checked by a unit test):"));
  out.push(...code([
    "def preprocess(self, image):",
    "    image = ImageOps.exif_transpose(image).convert(\"RGB\")",
    "    w, h = image.size                      # resize shorter side to 224 (as torchvision.Resize)",
    "    new_w, new_h = (224, int(224 * h / w)) if w <= h else (int(224 * w / h), 224)",
    "    image = image.resize((new_w, new_h), BICUBIC)",
    "    top, left = round((new_h - 224) / 2), round((new_w - 224) / 2)",
    "    image = image.crop((left, top, left + 224, top + 224))   # centre crop",
    "    arr = np.asarray(image, dtype=np.float32) / 255.0",
    "    arr = (arr - self.mean) / self.std     # ImageNet normalisation",
    "    return arr.transpose(2, 0, 1)          # H x W x C -> C x H x W",
  ]));
  out.push(p("Predictions use ONNX Runtime and temperature-scaled soft-max probabilities:"));
  out.push(...code([
    "logits = self.session.run(None, {self.input_name: batch})[0]",
    "z = logits / self.temperature             # calibration fitted on the validation set",
    "probs = np.exp(z - z.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)",
    "prediction.is_confident = probs.max() >= threshold",
  ]));
  out.push(...figure("app_single", R.figApp("app_single.png"), 15.9, "The Classify-an-Item Tab with a Confident Prediction",
    "An uploaded photo of a plastic bottle (test image) is recognised with high calibrated confidence; the card shows the bin, recycling tips and the probability of every class."));
  out.push(p(`When the confidence is below the threshold, the application does not pretend to know: it shows an *UNCERTAIN* warning with the second most likely class and phrases the disposal advice conditionally (${figRef("app_uncertain")}). The example is one of the hardest test images: a packet of disposable face masks labelled *trash*, for which the model hesitates between cardboard (39.7%) and trash (38.5%).`));
  out.push(...figure("app_uncertain", R.figApp("app_uncertain_main.png"), 14, "An Uncertain Prediction",
    "Confidence 39.7% is below the 60% threshold, so the result is flagged, the second most likely class is named and the user is asked to check the item."));
  out.push(...figure("app_batch", R.figApp("app_batch_main.png"), 15.9, "Batch Classification of a Folder With Valid and Invalid Files",
    "Summary counts, per-file table (confidence as bars), a full explanation for every rejected or flagged file, CSV download and the number of items per predicted material."));
  out.push(...figure("app_invalid", R.figApp("app_invalid_main.png"), 14, "Validation of an Invalid File",
    "A damaged JPEG file (truncated download) is rejected with an explanation instead of crashing the application."));
  return out;
}

function section32(R) {
  const out = [];
  out.push(h2("3.2 User Guide"));
  out.push(h3("What the application does"));
  out.push(p("Smart Waste Sorter looks at a photo of **one** waste item and predicts its material – cardboard, glass, metal, paper, plastic or trash (non-recyclable) – together with how sure it is and which bin to use."));
  out.push(h3("System requirements"));
  out.push(...bullets([
    "Windows 10/11, macOS 12+ or Linux; any modern browser (Chrome, Edge, Firefox, Safari);",
    "**Python 3.10 – 3.13** (python.org; on Windows tick *Add python.exe to PATH* during installation);",
    "internet connection the first time only (to install the libraries, ≈ 200 MB); about 400 MB of disk space; no GPU needed.",
  ]));
  out.push(h3("Installing and starting the application"));
  out.push(...numbered([
    "Extract the zip archive of Appendix A (`smart-waste-sorter-code.zip`) to a folder, e.g. *Documents\\smart-waste-sorter*.",
    "**Windows:** double-click `run_app.bat`. **macOS / Linux:** open a terminal in the folder and run `bash run_app.sh`.",
    "The first start creates a private environment (`.venv`) and installs the libraries (1–3 minutes). Later starts take a few seconds.",
    "The browser opens **http://localhost:8501**. If it does not, copy the *Local URL* printed in the console window into the browser.",
    "To stop the application close the console window (Windows) or press **Ctrl + C** in the terminal.",
  ]));
  out.push(p("*Manual alternative (any system)*, from the project folder:"));
  out.push(...code([
    "python -m venv .venv",
    ".venv\\Scripts\\activate          (macOS/Linux: source .venv/bin/activate)",
    "pip install -r requirements.txt",
    "streamlit run app/streamlit_app.py",
  ]));
  out.push(p("*Command-line version* (no browser): `python app/cli.py` shows a menu – 1) classify one image, 2) classify a folder, 3) change the threshold, 4) help, 5) quit. It can also be used directly, e.g. `python app/cli.py sample_images --csv results.csv`."));
  out.push(h3("Using the web application"));
  out.push(...table("guide_tabs", "Parts of the Web Application",
    ["Part", "How to use it"], [
      ["Sidebar", "**Confidence threshold** (default 60%): predictions below it are labelled UNCERTAIN. **Show the probability of every class**: all six classes instead of the top three. Model accuracy is shown below."],
      ["Tab 1 – Classify an item", "Choose *Upload a photo* (click *Browse files* or drag an image into the box), *Take a photo with the webcam* (allow camera access, click *Take photo*) or *Use a sample image*. The result appears immediately."],
      ["Tab 2 – Batch classification", "Either upload several photos and click *Classify uploaded images*, or type a folder path (e.g. C:\\Users\\me\\Pictures\\waste or /Users/me/waste) and click *Classify folder*. Click *Download results as CSV* to save the table."],
      ["Tab 3 – Model performance", "Test-set accuracy, macro-F1, top-2 accuracy, ROC-AUC, per-class scores and the confusion matrix."],
      ["Tab 4 – Help", "Short version of this guide."],
    ], [26, 74], null, { size: 17 }));
  out.push(h3("Input format"));
  out.push(...bullets([
    "Image files: **.jpg, .jpeg, .png, .bmp, .webp, .tif, .tiff**, up to 20 MB, at least 32 × 32 pixels (any size above is fine; the app resizes it).",
    "Content: one item, ideally filling most of the frame, well lit, on a plain background (like the training photos).",
    "Folders: all supported images directly inside the folder (sub-folders are ignored), maximum 500 per run.",
    "Unsupported, empty, corrupted, tiny or extremely elongated files are **rejected** with a message; grey-scale, transparent, animated or low-resolution images are **accepted with a warning**.",
  ]));
  out.push(h3("Meaning of the outputs"));
  out.push(...table("outputs", "Outputs Produced by the Application",
    ["Output", "Meaning"], [
      ["Prediction", "The most likely material (one of the six classes)."],
      ["Confidence", "Calibrated probability (0–100%) that the prediction is correct. With the default threshold of 60%, " + R.coverage_pct + " of test images were accepted and " + R.acc_above_pct + " of those were correct."],
      ["Status", "**OK** = confidence ≥ threshold; **UNCERTAIN** = below threshold – check the item manually or take a clearer photo; **REJECTED** = invalid input (see message)."],
      ["Second choice", "The next most likely class and its probability (useful for uncertain cases)."],
      ["Where does it go? / Disposal", "Bin under New Zealand's standard kerbside rules plus recycling tips. Bin colours vary by council."],
      ["Probability table / bar", "Probability of every class; the values add up to 100%."],
      ["CSV file", "Columns File, Prediction, Confidence (0–1), Second choice, Status, Disposal, Message – one row per file."],
    ], [24, 76], null, { size: 17 }));
  out.push(h3("Troubleshooting"));
  out.push(...table("trouble", "Common Problems and Solutions", ["Problem", "Solution"], [
    ["\"Python 3.10 or newer is required\"", "Install Python from python.org (tick *Add python.exe to PATH*) and run the launcher again."],
    ["Installation of libraries fails", "Check the internet connection and run the launcher again; it resumes the installation."],
    ["Browser does not open", "Open http://localhost:8501, or the Local URL printed in the console."],
    ["Port 8501 already in use", "Streamlit chooses the next free port (8502, …); use the URL shown in the console."],
    ["\"The model could not be loaded\"", "Keep the folders app and model together and do not rename the files inside model."],
    ["Webcam option shows no camera", "Allow camera access in the browser, or upload a photo instead."],
  ], [34, 66], null, { size: 17 }));
  out.push(h3("Libraries used"));
  out.push(...table("libraries", "Python Libraries Used by the Project",
    ["Library", "Version tested", "Used for"], R.libraryRows, [24, 18, 58],
    "The application itself only needs the first five libraries (requirements.txt). The others were used for data analysis and training (requirements-train.txt).", { size: 17 }));
  return out;
}

function conclusion(R) {
  const trash = R.trash_sources ? Object.keys(R.trash_sources).slice(0, 4).join(", ") : null;
  return [
    h1("4. Conclusion"),
    p(`This project developed an intelligent system that classifies photos of waste into six material classes and turns the prediction into disposal advice. The data analysis went beyond describing the data set: it traced the origin of the images and discovered that ${R.exact_dup_pct} of the files belong to byte-identical pairs, which, if ignored, would leak test images into training and overstate performance. After cleaning and a leakage-free stratified group split, an EfficientNet-B0 network fine-tuned with transfer learning achieved **${R.acc_pct} accuracy and ${R.f1_3} macro-F1** on unseen test images, clearly outperforming classical machine-learning models (≤ ${R.best_classical_acc}) and a CNN trained from scratch (${R.cnn_acc}). The model is calibrated, its decisions can be inspected with Grad-CAM, it is robust to common photo imperfections except ${R.robust_weak}, and it runs in about ${R.latency_ms} ms per image on an ordinary CPU inside a validated, user-friendly Streamlit application. Development was iterative: the evaluation of the first training run exposed weaknesses (an inefficient first training phase and fragility to noisy photos) that were fixed in the final model.`),
    p(`**Limitations.** The model only knows six classes: objects of other types or photos with several items are forced into the closest class, which the uncertainty flag only partly mitigates. The classes were assembled from different source collections with different photographic styles, so part of what the model learns may be the *source* rather than the material, and performance on everyday kitchen photos may be lower than on the test set. The labels follow the data set rather than New Zealand's recycling rules: the *trash* class mixes very different items${trash ? ` (mainly ${trash})` : ""}, and drink cartons are labelled *cardboard* although they are not accepted at kerbside – the application therefore adds rule-based advice to the prediction. Plastic identification is limited to the material; the resin number (only 1, 2 and 5 are accepted) cannot be read from a typical photo.`),
    p("**Future work.** Collecting local New Zealand photos with labels that follow the kerbside rules (including drink cartons, soft plastics and resin codes), adding an *other* class for unknown items, object detection for photos with several items, testing on photos from a different source, and a quantised model for mobile phones would make the system closer to real-world deployment in homes or in smart bins."),
  ];
}

function references() {
  const refs = [
    "Auckland Council. (2024, January). *New standards to recycle right*. OurAuckland. https://ourauckland.aucklandcouncil.govt.nz/news/2024/01/new-standards-to-recycle-right/",
    "Anthropic. (2026). *Claude* (Opus 5.5 version) [Large language model]. https://claude.ai",
    "Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324",
    "Canny, J. (1986). A computational approach to edge detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8*(6), 679–698. https://doi.org/10.1109/TPAMI.1986.4767851",
    "Cofone, L. (2025). *Garbage images dataset (2000/class)* (Version 5) [Data set]. Kaggle. https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification",
    "Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297. https://doi.org/10.1007/BF00994018",
    "Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. In *2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition* (Vol. 1, pp. 886–893). IEEE. https://doi.org/10.1109/CVPR.2005.177",
    "Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 248–255). IEEE. https://doi.org/10.1109/CVPR.2009.5206848",
    "Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations*. https://openreview.net/forum?id=YicbFdNTTy",
    "Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.",
    "Google. (2026). *Google Colaboratory* [Computer software]. https://colab.research.google.com",
    "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321–1330). PMLR.",
    "Hasler, D., & Süsstrunk, S. (2003). Measuring colourfulness in natural images. In *Proceedings of SPIE 5007, Human Vision and Electronic Imaging VIII* (pp. 87–95). https://doi.org/10.1117/12.477378",
    "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90",
    "Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 7132–7141). https://doi.org/10.1109/CVPR.2018.00745",
    "Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Article 15. https://doi.org/10.1145/2382577.2382579",
    "Kaza, S., Yao, L. C., Bhada-Tata, P., & Van Woerden, F. (2018). *What a waste 2.0: A global snapshot of solid waste management to 2050*. World Bank. https://doi.org/10.1596/978-1-4648-1329-0",
    "Kunwar, S. (2026). *The Garbage Dataset (GD): A multi-class image benchmark for automated waste segregation* (arXiv:2602.10500). arXiv. https://doi.org/10.48550/arXiv.2602.10500",
    "Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. In *International Conference on Learning Representations*. https://openreview.net/forum?id=Skq89Scxx",
    "Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations*. https://openreview.net/forum?id=Bkg6RiCqY7",
    "Mao, W.-L., Chen, W.-C., Wang, C.-T., & Lin, Y.-H. (2021). Recycling waste classification using optimized convolutional neural network. *Resources, Conservation and Recycling, 164*, Article 105132. https://doi.org/10.1016/j.resconrec.2020.105132",
    "Ministry for the Environment. (2023). *Standard materials for kerbside collections: Guidance for territorial authorities*. New Zealand Government. https://environment.govt.nz/publications/standard-materials-for-kerbside-collections-guidance-for-territorial-authorities/",
    "Mohamed, M. (2021). *Garbage classification (12 classes)* [Data set]. Kaggle. https://www.kaggle.com/datasets/mostafaabla/garbage-classification",
    "Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 24*(7), 971–987. https://doi.org/10.1109/TPAMI.2002.1017623",
    "ONNX Runtime developers. (2026). *ONNX Runtime* [Computer software]. https://onnxruntime.ai",
    "Parker, D. (2023, March 29). *Standard kerbside recycling part of new era for waste system* [Press release]. New Zealand Government. https://www.beehive.govt.nz/release/standard-kerbside-recycling-part-new-era-waste-system",
    "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32* (pp. 8024–8035). Curran Associates.",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.",
    "Pech-Pacheco, J. L., Cristóbal, G., Chamorro-Martínez, J., & Fernández-Valdivia, J. (2000). Diatom autofocusing in brightfield microscopy: A comparative study. In *Proceedings of the 15th International Conference on Pattern Recognition* (Vol. 3, pp. 314–317). IEEE. https://doi.org/10.1109/ICPR.2000.903548",
    "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 4510–4520). https://doi.org/10.1109/CVPR.2018.00474",
    "Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74",
    "Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data, 6*, Article 60. https://doi.org/10.1186/s40537-019-0197-0",
    "Snowflake Inc. (2026). *Streamlit* [Computer software]. https://streamlit.io",
    "Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception architecture for computer vision. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 2818–2826). https://doi.org/10.1109/CVPR.2016.308",
    "Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the 36th International Conference on Machine Learning* (Vol. 97, pp. 6105–6114). PMLR.",
    "Thung, G., & Yang, M. (2016). *Classification of trash for recyclability status* [Course project report, CS229 Machine Learning]. Stanford University. https://github.com/garythung/trashnet",
    "van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research, 9*(86), 2579–2605.",
    "Wightman, R. (2019). *PyTorch image models* [Computer software]. GitHub. https://doi.org/10.5281/zenodo.4414861",
    "Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? In *Advances in Neural Information Processing Systems 27* (pp. 3320–3328). Curran Associates.",
    "Zauner, C. (2010). *Implementation and benchmarking of perceptual image hash functions* [Master's thesis, Upper Austria University of Applied Sciences].",
  ];
  refs.sort((a, b) => a.localeCompare(b, "en", { sensitivity: "base" }));   // APA: alphabetical by author
  return [h1("References"), ...refs.map(reference)];
}

function appendices(R) {
  return [
    new Paragraph({ children: [new PageBreak()] }),
    h1("Appendices"),
    h2("Appendix A – Source Code (zip archive)"),
    p(`**File:** \`${R.zip_code}\` (embedded below). It contains the complete project: the application (\`app/\`), the trained model (\`model/\`), sample images, the analysis and training package (\`src/waste_classifier/\`), the Colab notebook (\`notebooks/Smart_Waste_Sorter_Colab.ipynb\`), scripts, unit tests, launchers, requirements files, README and user guide. To run the application see Section 3.2.`),
    p("[EMBEDDED FILE: " + R.zip_code + "]", { run: { bold: true, color: "1E5B3A" } }),
    h2("Appendix B – Data and Model Files (zip archive)"),
    p(`**File:** \`${R.zip_data}\` (embedded below). It contains the data and model files produced by the Colab notebook: the trained model (\`model/waste_classifier.onnx\` and \`model_info.json\` with class names, normalisation constants, calibration temperature and test metrics), all metric tables (\`metrics/\`: per-image statistics and duplicate information \`image_stats.csv\`, the train/validation/test assignment of every image \`splits.csv\`, provenance matches, baseline results, training histories, test-set predictions, per-class metrics, calibration and robustness results) and the sample images (the figures are not repeated because they are all in this report). The ${R.n_images_fmt} original images (≈121 MB) are not duplicated inside the archive, to keep the report within the upload limit; they are published under the MIT licence at https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification (Cofone, 2025), and \`splits.csv\` identifies exactly which files were used in each subset.`),
    p("[EMBEDDED FILE: " + R.zip_data + "]", { run: { bold: true, color: "1E5B3A" } }),
    h2("Appendix C – Declaration of the Use of Artificial Intelligence Tools"),
    p("In line with the course policy on AI use, the following tools were used and are referenced (APA 7):"),
    ...bullets([
      "**Claude (Anthropic, 2026)** was used as a programming and writing assistant: to help structure the project, draft and review Python code (analysis pipeline, training and evaluation code, Streamlit application, tests), design visualisations and draft parts of the report text. All code was executed, tested and checked by the student, the results were produced by running the notebook on the data, and the analysis and conclusions were reviewed and edited by the student.",
      "**Google Colaboratory (Google, 2026)** provided the GPU environment used for training.",
      "No other generative AI tools were used.",
    ]),
    p("*Student: review this declaration and adjust it so that it describes exactly how you used AI tools.*", { run: { color: "8A4B08", size: 19 } }),
    h2("Appendix D – Team Agreement, Meeting Minutes and Peer Evaluation"),
    p("Attach the completed forms from Appendices A–C of the project brief (Team Agreement/Contract, Meeting Minutes and Peer Evaluation) here if the project was completed as a group of two. Delete this section if the project was completed individually and the lecturer does not require the forms."),
  ];
}

module.exports = { section31, section32, conclusion, references, appendices };

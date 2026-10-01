// Sections 2.1 (model choice), 2.2 (data transformation) and 2.3 (implementation)
const L = require("./lib");
const { p, h1, h2, h3, bullets, numbered, figure, table, code, figRef, tabRef } = L;

function section21(R) {
  const out = [];
  out.push(h1("2. Problem Modelling"));
  out.push(h2("2.1 Description of and Justification for the Modelling Technique"));
  out.push(p("The task is **supervised multi-class image classification**: learn a function *f(image) → class* from labelled examples, where the output is one of six mutually exclusive materials. Several families of models can solve this kind of problem; they differ in how the image is turned into features, how much data and computing power they need, and how well they generalise. The main options are compared in " + tabRef("model_options") + "."));
  out.push(...table("model_options", "Candidate Modelling Techniques for Waste-Image Classification",
    ["Technique", "How it works", "Advantages", "Disadvantages for this problem"],
    [
      ["k-nearest neighbours, logistic regression, SVM, random forest on hand-crafted features", "Each image is summarised by engineered features – colour histograms, HOG edge-orientation histograms (Dalal & Triggs, 2005), LBP texture codes (Ojala et al., 2002) – that a classical classifier separates (Breiman, 2001; Cortes & Vapnik, 1995)", "Fast, little data needed, well understood, some interpretability", "Features are fixed by the designer and cannot describe material cues such as transparency, gloss or crumpling; sensitive to background, pose and lighting"],
      ["Multilayer perceptron on raw pixels", "Every pixel is an input to fully connected layers", "Simple", "256 × 256 × 3 = 196,608 inputs → millions of weights; no notion of spatial locality; over-fits badly"],
      ["Convolutional neural network (CNN) trained from scratch", "Stacked convolution filters learn local patterns, pooling builds position invariance, the last layers classify", "Learns features automatically; strong inductive bias for images", "Needs much more data and training time than available here to learn good general features; prone to over-fitting"],
      ["**Transfer learning with a pre-trained CNN** (VGG, ResNet, DenseNet, MobileNet, EfficientNet)", "A CNN pre-trained on 1.28 million ImageNet photos (Deng et al., 2009) is reused as a feature extractor and fine-tuned on waste images (Yosinski et al., 2014)", "High accuracy with thousands (not millions) of images; generic edge/texture/shape features transfer well; widely used for waste sorting (Mao et al., 2021)", "Larger models are slow on a CPU; pre-trained input format must be respected"],
      ["Vision Transformer (ViT)", "Image split into patches processed by self-attention (Dosovitskiy et al., 2021)", "State-of-the-art accuracy with very large pre-training", "Much heavier (≈86 M parameters for ViT-B/16), data-hungry, slower on the lecturer's CPU"],
    ], [20, 28, 24, 28], "HOG = histogram of oriented gradients; LBP = local binary patterns; SVM = support vector machine.", { size: 16 }));

  out.push(h3("Chosen model: EfficientNet-B0 with transfer learning"));
  out.push(p("The chosen technique is **transfer learning with EfficientNet-B0** (Tan & Le, 2019), a convolutional neural network pre-trained on ImageNet and fine-tuned on the waste images. EfficientNet was designed with *compound scaling*: instead of making a network only deeper (like ResNet-152) or wider, depth, width and input resolution are scaled together in a fixed ratio found by neural architecture search, giving better accuracy for the same computational cost. B0 is the smallest member of the family."));
  out.push(p("Its structure is:"));
  out.push(...bullets([
    "a 3 × 3 convolution stem (32 filters, stride 2) that turns the 224 × 224 × 3 image into feature maps;",
    "16 **MBConv** blocks in 7 stages – inverted residual bottlenecks with depthwise-separable convolutions (Sandler et al., 2018), 3 × 3 or 5 × 5 kernels, **squeeze-and-excitation** channel attention (Hu et al., 2018) and the smooth SiLU (Swish) activation – which progressively reduce the spatial size to 7 × 7 while increasing the number of channels to 320;",
    "a 1 × 1 convolution to 1,280 channels followed by **global average pooling**, producing a 1,280-number description of the image;",
    "**dropout** (p = 0.3) and a new **fully connected layer with six outputs** whose soft-max gives the probability of each material. This layer replaces the original 1,000-class ImageNet classifier.",
  ]));
  out.push(p(`In total the network has ${R.n_params_fmt} trainable parameters and needs about 0.39 billion floating-point operations per image, compared with 25.6 million parameters and 4.1 billion operations for ResNet-50 (He et al., 2016), while reaching a higher ImageNet top-1 accuracy (77.7% vs 76.1% for the weights used; Wightman, 2019).`));
  out.push(p("**Why this model is suitable for this problem:**"));
  out.push(...bullets([
    "**Data efficiency.** ImageNet contains many object categories relevant to waste (bottles, cans, cartons, boxes, bags), so the early and middle layers already detect edges, textures, reflections and shapes. Only the adaptation to six materials has to be learnt from our ≈8,000 training images, which avoids the over-fitting of a network trained from scratch.",
    "**Accuracy.** Pre-trained CNNs are the established state of the art for material and waste classification (Mao et al., 2021), and the baselines in this project confirm it empirically (" + tabRef("baselines") + ").",
    "**Efficiency for deployment.** B0 is small (≈" + R.onnx_mb + " MB as an ONNX file) and fast on an ordinary CPU (≈" + R.latency_ms + " ms per image), so the application runs on the lecturer's computer without a GPU – a stated requirement of the project.",
    "**Explainability and calibration.** Its convolutional feature maps support Grad-CAM heat-maps (Selvaraju et al., 2017), and its outputs can be calibrated with temperature scaling (Guo et al., 2017) so that the confidence shown to the user is meaningful.",
    "**Compared with the alternatives,** it is far more accurate than classical models on hand-crafted features, less data-hungry than a CNN from scratch, and lighter and faster than VGG/ResNet-50 or a Vision Transformer with little or no loss of accuracy.",
  ]));
  out.push(p(`${figRef("pipeline")} shows where the model sits in the complete system: it is trained and evaluated in a Colab notebook, exported once, and then used by the application on the user's computer.`));
  out.push(...figure("pipeline", R.figStatic("fig00_pipeline.png"), 15.5, "End-to-End Pipeline of the Smart Waste Sorter",
    "Steps 1–6 run in the Colab notebook (Appendix A); steps 7–8 are the application delivered to the user (Section 3)."));
  out.push(p(`To support the choice with evidence rather than only literature, six alternative models were trained **on exactly the same training data and evaluated on the same test set** (${tabRef("baselines")}). The classical models reached ${R.classical_range} test accuracy, the CNN trained from scratch ${R.cnn_acc}, and a linear classifier on *frozen* ImageNet features already ${R.probe_acc} – showing the value of pre-trained features. Fine-tuning the whole EfficientNet-B0 gave the best result, ${R.acc_pct}.`));
  out.push(...table("baselines", "Comparison of Modelling Techniques on the Same Train/Validation/Test Split",
    ["Model", "Family", "Validation accuracy", "Test accuracy", "Test macro-F1", "Training time"],
    R.baselineRows, [34, 20, 12, 11, 11, 12],
    "All models used the de-duplicated stratified group split described in Section 2.2. Training times on a Google Colab T4 GPU (classical models on CPU). The final model is highlighted.",
    { size: 16, highlightRow: R.baselineRows.length - 1 }));
  return out;
}

function section22(R) {
  const out = [];
  out.push(h2("2.2 Data Transformation"));
  out.push(p("The raw JPEG files cannot be given directly to the network: they contain duplicates, they are 256 × 256 pixels with values 0–255, and the pre-trained EfficientNet expects 224 × 224 tensors normalised in the same way as its ImageNet training data. The following transformations were applied, in this order (" + figRef("pipeline") + ")."));
  out.push(h3("1. Cleaning and de-duplication"));
  out.push(p(`Exact duplicates were detected with an MD5 hash of the file bytes. Within each class only one copy was kept (**${R.removed_exact_fmt} files removed**); files whose identical copies carried *different* labels are ambiguous and were removed completely (**${R.removed_conflicts_fmt} files**). *Why:* duplicated images add no information, bias the model toward repeated photos and, most importantly, leak test images into training. After cleaning, **${R.n_model_fmt} images** remained.`));
  out.push(h3("2. Label encoding"));
  out.push(p("Class names were mapped to integers in alphabetical order (cardboard = 0, glass = 1, metal = 2, paper = 3, plastic = 4, trash = 5). *Why:* the cross-entropy loss and the evaluation metrics work with integer class indices; the six output neurons of the network correspond to these indices, and the mapping is stored in `model_info.json` so that the application translates predictions back into names."));
  out.push(h3("3. Stratified group split into training, validation and test sets"));
  out.push(p(`Near-duplicate images (perceptual-hash distance ≤ 4) were linked into groups with a union-find algorithm (${R.dup_groups_multi_fmt} groups with more than one image). A **stratified group split** (scikit-learn \`StratifiedGroupKFold\`, seed 42) then assigned about 70% of the images to training, 15% to validation and 15% to testing, such that (a) every subset keeps the class proportions of the full data set and (b) all images of a group fall into the same subset. ${tabRef("split")} and ${figRef("split")} show the result; a check confirmed that **no duplicate group spans two subsets**. *Why:* the validation set is used to choose the epoch, the temperature and to compare models, while the untouched test set gives an unbiased estimate of performance on new photos; grouping prevents near-identical photos from inflating the test score.`));
  out.push(...table("split", "Number of Images per Class in Each Subset after Cleaning",
    ["Class", "Training", "Validation", "Test", "Total"], R.splitRows, [28, 18, 18, 18, 18],
    null, { size: 17, boldRow: R.splitRows.length - 1 }));
  out.push(...figure("split", R.fig("fig13_split_distribution.png"), 13, "Class Distribution in the Training, Validation and Test Sets",
    "The three subsets have almost identical class proportions (stratification)."));
  out.push(h3("4. Resizing and cropping"));
  out.push(p("Each image is resized so that its shorter side is 224 pixels (bicubic interpolation) and the central 224 × 224 square is kept. For the square 256 × 256 data set images this is simply a down-scaling of the whole image; for photos with other shapes uploaded in the application, the centre of the photo is used without distorting it. *Why:* 224 × 224 is EfficientNet-B0's native input resolution, the resolution at which its pre-trained filters work best, and a fixed size is needed to form mini-batches."));
  out.push(h3("5. Conversion to tensors and normalisation"));
  out.push(p("Pixel values are converted from integers 0–255 to floating-point numbers in [0, 1], rearranged from height × width × channel to channel × height × width, and each colour channel is standardised with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225): *x' = (x − μ) / σ*. *Why:* the pre-trained weights expect inputs with exactly this distribution; zero-centred, unit-variance inputs also keep activations and gradients in a stable range during fine-tuning."));
  out.push(h3("6. Data augmentation (training set only)"));
  out.push(p(`In every epoch each training image is randomly transformed before it is used: random-resized crop (60–100% of the area, aspect ratio 0.8–1.25), horizontal flip (probability 0.5), vertical flip (0.2), rotation of up to ±20°, brightness, contrast and saturation jitter of ±25%, Gaussian blur (probability 0.2, σ = 0.1–1.5 pixels) and Gaussian sensor noise (probability 0.3, standard deviation up to 0.08 of the intensity range) (${figRef("augmentation")}). *Why:* waste can be photographed from any angle, distance and lighting, and phone photos are often slightly blurred or noisy, so these variations are plausible; showing the network many versions of each photo acts as regularisation and reduces over-fitting (Shorten & Khoshgoftaar, 2019). The **hue is deliberately not changed**, because colour carries class information (brown cardboard, green or brown glass; Section 1.2.3). Blur and noise were added after the first training run showed that the model was fragile to them (see "Iterative improvement" in Section 2.3). Validation and test images are **not** augmented, so evaluation reflects real, unmodified photos.`));
  out.push(...figure("augmentation", R.fig("fig14_augmentation_examples.png"), 14, "Examples of Training-Time Data Augmentation",
    "Each row shows one training image (left) and five random augmented versions produced by the transformation pipeline."));
  out.push(h3("7. Mini-batching"));
  out.push(p("Transformed images are grouped into shuffled mini-batches of 64 by a PyTorch `DataLoader` using two worker processes. *Why:* mini-batch gradient descent is required for training on a GPU, and shuffling each epoch avoids order effects."));
  out.push(h3("8. Transformations for the comparison models"));
  out.push(p("The classical baselines need fixed-length feature vectors instead of tensors, so for them each image was resized to 128 × 128, converted to grey-scale for the HOG (1,764 values) and LBP texture (10 values) descriptors and to HSV for a colour histogram (128 values). The resulting 1,902 features were **standardised** (z-scores) and reduced with **principal component analysis** to 150 components before k-NN, logistic regression and SVM; *why:* distance- and margin-based models are sensitive to feature scale, and PCA removes redundant, noisy dimensions."));
  out.push(h3("9. The same transformation in the application"));
  out.push(p("The application re-implements the evaluation transform (steps 4–5) with NumPy and Pillow, so that PyTorch is not needed on the lecturer's computer. A unit test checks that the application's pre-processing matches the PyTorch transform used during training (maximum difference < 0.0001), guaranteeing that the deployed model receives exactly the kind of input it was evaluated on."));
  return out;
}

function section23(R) {
  const out = [];
  out.push(h2("2.3 Implementation of the Chosen Technique"));
  out.push(p(`The model was implemented in Python ${R.env.python} with PyTorch ${R.env.torch} (Paszke et al., 2019) and the \`timm\` model library ${R.env.timm} (Wightman, 2019), on Google Colab with an NVIDIA ${R.env.gpu} GPU. All code lives in a documented package, \`waste_classifier\` (Appendix A); the Colab notebook runs the steps in order and can reproduce every figure and number in this report with a fixed random seed (42); the overall pipeline is shown in ${figRef("pipeline")}.`));
  out.push(h3("Steps to construct the model"));
  out.push(...numbered([
    "**Load the data** – the Kaggle archive is extracted and indexed (path, class, label) and cross-checked with `metadata.csv`.",
    "**Analyse and clean** – image statistics, integrity checks and duplicate detection (Section 1).",
    "**Split and transform** – stratified group split and the transformations of Section 2.2 (`data.py`).",
    "**Create the network** – `timm.create_model(\"efficientnet_b0\", pretrained=True, num_classes=6, drop_rate=0.3, drop_path_rate=0.1)` loads the ImageNet weights and replaces the classifier with a new 6-output layer whose weights are initialised to **zero**, so that training starts from uniform predictions (loss = ln 6) instead of large random scores; stochastic depth (drop-path) randomly skips residual blocks during training as extra regularisation.",
    "**Phase 1 – train the head (" + R.cfg.head_epochs + " epochs).** All convolutional layers are frozen; only the new classifier (" + R.head_params_fmt + " parameters) is trained with AdamW (learning rate " + R.cfg.head_lr + "). *Why:* the randomly initialised head would otherwise send large, noisy gradients into the pre-trained layers and damage them.",
    "**Phase 2 – fine-tune all layers (up to " + R.cfg.finetune_epochs + " epochs).** Every layer is unfrozen and trained with a ten-times smaller peak learning rate (" + R.cfg.finetune_lr + ") so that the pre-trained features are adapted gently to waste images.",
    "**Optimisation details** – loss: cross-entropy with label smoothing 0.1 (Szegedy et al., 2016), which discourages over-confident predictions; optimiser: AdamW with decoupled weight decay 0.01 (Loshchilov & Hutter, 2019); learning-rate schedule: linear warm-up then cosine decay, updated every batch (Loshchilov & Hutter, 2017); automatic mixed precision (float16) on the GPU for speed.",
    "**Model selection with early stopping** – after every epoch the model is evaluated on the validation set; the weights with the best validation macro-F1 are kept and training stops after " + R.cfg.patience + " epochs without improvement.",
    "**Calibration** – a single temperature *T* is fitted on the validation set by minimising the negative log-likelihood (temperature scaling; Guo et al., 2017). Dividing the logits by *T* = " + R.temperature_2 + " makes the reported confidence match the observed accuracy without changing any prediction.",
    "**Evaluation** – the selected model is applied once to the test set (Section 2.4).",
    "**Export** – the network is exported to the ONNX format (opset 17) with a dynamic batch dimension; large weight tensors are stored as float16 to halve the file (" + R.onnx_mb + " MB). The ONNX outputs were compared with PyTorch on " + R.parity_n + " test images: predictions agreed for " + R.parity_agree + " of them (maximum logit difference " + R.parity_diff + ").",
  ]));
  out.push(...table("hyper", "Final Model Configuration and Hyper-parameters",
    ["Setting", "Value", "Reason"], [
      ["Architecture", "EfficientNet-B0, ImageNet weights (timm)", "Best accuracy/efficiency trade-off (Section 2.1)"],
      ["Input", "224 × 224 × 3, ImageNet normalisation", "Native resolution of the pre-trained model"],
      ["Output", "6 logits → soft-max (temperature " + R.temperature_2 + ")", "One probability per class, calibrated"],
      ["Batch size", String(R.cfg.batch_size), "Fits the T4 GPU memory with mixed precision"],
      ["Epochs", `${R.cfg.head_epochs} (head) + up to ${R.cfg.finetune_epochs} (fine-tune); best epoch ${R.best_epoch}`, "Early stopping, patience " + R.cfg.patience],
      ["Optimiser", `AdamW, lr ${R.cfg.head_lr} / ${R.cfg.finetune_lr}, weight decay ${R.cfg.weight_decay}`, "Stable fine-tuning of pre-trained weights"],
      ["Schedule", "Warm-up + cosine decay (per batch)", "Large steps early, small steps at the end"],
      ["Regularisation", "Dropout 0.3, drop-path 0.1, label smoothing 0.1, augmentation", "Reduce over-fitting on ≈8,000 images"],
      ["Selection metric", "Validation macro-F1", "Treats all six classes equally"],
      ["Training time", R.train_time, "Google Colab, NVIDIA T4"],
    ], [22, 40, 38], null, { size: 17 }));
  out.push(p(`${figRef("learning")} shows the learning curves. ${R.learning_text}`));
  out.push(...figure("learning", R.fig("fig15_learning_curves.png"), 15, "Training and Validation Loss and Accuracy per Epoch",
    "The vertical line marks the start of phase 2 (fine-tuning of all layers). Training metrics are computed on augmented images and with dropout active, which is why they can be lower than the validation metrics."));
  if (R.iteration) {
    out.push(h3("Iterative improvement of the model"));
    out.push(p(R.iteration.text));
    out.push(...table("iteration", "First and Final Training Run Compared", ["Measure", "Run 1", "Run 2 (final)", "Change"],
      R.iteration.rows, [40, 18, 20, 22], R.iteration.note, { size: 17 }));
  }
  out.push(p("The core of the training loop is shown below (simplified from `train.py`):"));
  out.push(...code([
    "for x, y in train_loader:                       # one mini-batch of augmented images",
    "    x, y = x.to(device), y.to(device)",
    "    with torch.autocast('cuda', dtype=torch.float16):   # mixed precision",
    "        logits = model(x)                       # forward pass -> 6 scores per image",
    "        loss = criterion(logits, y)             # cross-entropy with label smoothing",
    "    optimizer.zero_grad()",
    "    scaler.scale(loss).backward()               # back-propagation",
    "    scaler.step(optimizer); scaler.update()     # AdamW weight update",
    "    scheduler.step()                            # warm-up / cosine learning rate",
    "# after each epoch: evaluate on validation set, keep the weights with best macro-F1",
  ]));
  return out;
}

module.exports = { section21, section22, section23 };

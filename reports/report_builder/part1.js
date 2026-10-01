// Cover page, front matter and Section 1.1 (Description and explanation)
const { Paragraph, TextRun, AlignmentType, PageBreak, TableOfContents } = require("docx");
const L = require("./lib");
const { p, h1, h2, h3, bullets, figure, table, figRef, tabRef } = L;

function cover(R) {
  const line = (text, opts = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: opts.after ?? 120 },
    children: [new TextRun({ text, size: opts.size || 24, bold: opts.bold, italics: opts.italics, color: opts.color || L.INK })] });
  return [
    new Paragraph({ children: [], spacing: { before: 1400 } }),
    line("Auckland Institute of Studies", { size: 26, bold: true, color: L.GREEN }),
    line("INFO813 Artificial Intelligence", { size: 24 }),
    line("Project – Trimester 3, 2026", { size: 24, after: 900 }),
    line("Smart Waste Sorter", { size: 56, bold: true, color: L.GREEN, after: 160 }),
    line("Classifying recyclable waste images with transfer learning", { size: 30, italics: true, after: 160 }),
    line("and an interactive Python application", { size: 30, italics: true, after: 1300 }),
    line("Student name: [Your full name]    Student ID: [Your ID]", { size: 24 }),
    line("Group member (if any): [Name]    Student ID: [ID]", { size: 24 }),
    line("Lecturer: [Lecturer's name]", { size: 24 }),
    line("Due date: Monday 30 November 2026, 4:00 pm", { size: 24, after: 600 }),
    line(`Word count (main text, excluding tables, figures, references and appendices): approx. ${R.wordCountLabel}`, { size: 20, italics: true, color: L.MUTED }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

function frontMatter(R) {
  return [
    new Paragraph({ children: [new TextRun({ text: "Table of Contents", bold: true, size: 32, color: L.GREEN })], spacing: { after: 200 } }),
    new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
    p("*If the table of contents is empty, right-click it in Word and choose Update Field.*", { run: { size: 18, color: L.MUTED } }),
    new Paragraph({ children: [new PageBreak()] }),
    h1("Executive Summary"),
    p(`This project builds an intelligent system that recognises the material of a piece of waste from a photograph – **cardboard, glass, metal, paper, plastic or trash** – and tells the user how to dispose of it under New Zealand's standard kerbside recycling rules. The data are the Kaggle *Garbage Images Dataset (2000/class)* (${R.n_images_fmt} RGB images of 256 × 256 pixels in six classes; Cofone, 2025). A critical analysis of the data showed that, although the data set is described as de-duplicated, **${R.exact_dup_files_fmt} images (${R.exact_dup_pct}) are byte-identical copies** of another image. Copies were removed and near-duplicates were kept together in one subset, leaving ${R.n_model_fmt} images split into training, validation and test sets without information leakage.`),
    p(`An **EfficientNet-B0** convolutional neural network pre-trained on ImageNet was fine-tuned with transfer learning (two-phase training, data augmentation, label smoothing and early stopping). On the held-out test set of ${R.n_test_fmt} images it reached an **accuracy of ${R.acc_pct}** (95% confidence interval ${R.acc_ci}) and a **macro-F1 of ${R.f1_3}**, far above the best classical baseline (${R.best_classical_name}, accuracy ${R.best_classical_acc}) and a CNN trained from scratch (${R.cnn_acc}). The model was exported to ONNX (${R.onnx_mb} MB) and embedded in a **Streamlit** web application – with a command-line alternative – that validates every input, classifies single photos, webcam pictures, several files or whole folders, reports calibrated confidence and flags uncertain predictions.`),
  ];
}

function section11(R) {
  const out = [];
  out.push(h1("1. Data Analysis"));
  out.push(h2("1.1 Description and Explanation"));

  out.push(h3("1.1.1 Topic and problem"));
  out.push(p("The topic of this project is **automatic classification of recyclable waste from images**. Given a colour photograph of a single discarded item, the system must decide which of six material categories it belongs to – cardboard, glass, metal, paper, plastic or non-recyclable trash – and use that decision to recommend the correct disposal route."));
  out.push(p(`The problem matters because sorting mistakes are expensive. Worldwide, cities generated 2.01 billion tonnes of municipal solid waste in 2016, a figure expected to reach 3.40 billion tonnes by 2050 (Kaza et al., 2018). In New Zealand, 16% of the material placed in household recycling bins cannot be recycled, while about 100,000 tonnes of recyclable material ends up in rubbish bins every year (Parker, 2023). Auckland Council reports that nearly a quarter of its kerbside recycling is contaminated, costing ratepayers an extra $3 million per year for sorting and disposal (Auckland Council, 2024). To reduce confusion, New Zealand standardised kerbside recycling on 1 February 2024: all councils now accept glass bottles and jars, paper and cardboard, plastic containers numbered 1, 2 and 5, and aluminium and steel cans (Ministry for the Environment [MfE], 2023). A tool that recognises the material from a photo and maps it to these rules directly addresses the "which bin?" problem at home, and the same model could drive sorting in smart bins or recycling facilities.`));

  out.push(h3("1.1.2 Why this is an appropriate problem and data set for an intelligent system"));
  out.push(p("Recognising a material from pixels is a perception task that cannot realistically be solved with hand-written rules, which is exactly the situation in which machine learning is appropriate:"));
  out.push(...bullets([
    "**No explicit rules exist.** Materials differ in subtle, combined cues – surface texture, reflectance, transparency, colour, printed patterns and shape. A crushed can, a brown glass bottle and a brown cardboard box can share colours; transparent plastic and glass look alike. An intelligent system must *learn* these cues from examples.",
    "**High variability.** Items appear deformed, dirty, partially visible, under different lighting and backgrounds and at different scales, so the system must generalise rather than memorise.",
    "**Labelled examples are available.** The data set provides thousands of images with a class label each, which suits supervised learning, and it is large enough to train and fairly evaluate a deep neural network.",
    "**The output is a decision with a real action.** Each class maps to a disposal instruction, so predictions can be embedded in software that supports people's everyday decisions – the core purpose of an intelligent system.",
    "**The difficulty is realistic and measurable.** Some classes are visually close (paper vs cardboard, plastic vs glass) and \"trash\" is a heterogeneous catch-all class, so the problem is neither trivial nor impossible; standard metrics (accuracy, precision, recall, F1) make progress objective.",
    "**It matches the course learning outcomes**: image pre-processing (LO2), selection of an appropriate AI algorithm (LO3), building and critically evaluating a model (LO4) and reporting a self-directed investigation (LO5). Waste classification is also explicitly listed as a preferred real-world application in the project brief.",
  ]));

  out.push(h3("1.1.3 Source of the data set"));
  out.push(p(`The data set was **downloaded from Kaggle** as the *Garbage Images Dataset (2000/class)*, version 5, published under the MIT licence by Leonardo Cofone (Kaggle user *zlatan599*; Cofone, 2025). According to its description, it was "made by a really big ensemble" of existing images, standardised to 256 × 256 RGB, cleaned of duplicates and balanced to roughly 2,300–2,500 images per class. The Kaggle "Provenance" field only points to a private notebook, so the original sources are not documented – a weakness for a data set used to train decision-making software.`));
  out.push(p(`To answer the question *"where did the data come from?"*, an empirical **provenance analysis** was carried out (notebook section 1, module \`provenance.py\`; details in Section 1.2.5): candidate public source data sets were downloaded and every image was compared with every image of our data set using 64-bit perceptual hashes (pHash; Zauner, 2010), which remain almost unchanged when an image is resized or re-compressed. ${R.provenance_text}`));
  out.push(p("The six-class taxonomy itself comes from **TrashNet** (Thung & Yang, 2016), the reference data set for this problem. It was photographed by two Stanford University students for a CS229 machine-learning project: each item was placed on a white poster board and photographed with iPhone 7 Plus, iPhone 5S and iPhone SE cameras in sunlight or room light, giving 2,527 images of 512 × 384 pixels. The other source collections consist of photographs of real waste items gathered from the web and from household environments. The data are therefore **real photographs of real objects** (not synthetic images), taken by different people with different cameras, backgrounds and lighting – an important property that shapes the analysis below."));

  out.push(h3("1.1.4 Description of the data set"));
  out.push(p(`${tabRef("dataset_summary")} summarises the data set and ${figRef("class_dist")} shows the number of images per class. Every file is a baseline JPEG image of 256 × 256 pixels in 8-bit RGB colour; the archive also contains \`metadata.csv\` (columns \`filename\`, \`label\`), which agrees 100% with the folder structure. ${figRef("samples")} shows random examples of each class.`));
  out.push(...table("dataset_summary", "Summary of the Garbage Images Dataset (version 5)",
    ["Property", "Value"],
    R.summaryRows, [42, 58],
    "Computed by the analysis pipeline (notebook Section 1). Near-duplicates: perceptual-hash Hamming distance ≤ 4 bits out of 64."));
  out.push(...figure("class_dist", R.fig("fig01_class_distribution.png"), 13.5,
    "Number of Images per Class",
    `The grey line marks the mean class size (${R.mean_class_fmt} images). The largest class (glass/trash, 2,500) is only ${R.imbalance_ratio} times larger than the smallest (metal, ${R.min_class_fmt}).`));
  out.push(...figure("samples", R.fig("fig02_sample_images.png"), 14.5,
    "Random Sample of Images from Each Class", "Six randomly chosen images per class (random seed 42)."));

  out.push(h3("1.1.5 Key properties: balance, quality and pre-processing requirements"));
  out.push(p(`**Class balance.** Judged by the file counts, the classes are close to balanced: shares range from ${R.share_min} to ${R.share_max}, the largest-to-smallest ratio is ${R.imbalance_ratio} and the normalised Shannon evenness is ${R.evenness} (1 = perfectly balanced; ${figRef("class_dist")}). The duplicate analysis below, however, shows that four classes are padded with copies (${figRef("dup_class")}): counted by *unique* photographs, the classes contain between ${R.dup.min_unique_fmt} (${R.dup.min_class}) and ${R.dup.max_unique_fmt} (${R.dup.max_class}) images, a ratio of ${R.dup.ratio_unique.toFixed(2)}. This moderate imbalance does not require re-sampling or class weights, but it is the reason why macro-averaged metrics, in which every class counts equally, are used for model selection and reporting.`));
  out.push(p(`**Data quality.** All ${R.n_images_fmt} files are readable and have identical size, format and colour mode, so no files had to be repaired. However, the analysis found a serious problem that the data set description does not mention: **${R.exact_dup_files_fmt} files (${R.exact_dup_pct}) form ${R.exact_dup_groups_fmt} pairs of byte-identical duplicates** (identical MD5 checksums), and ${R.near_dup_images_fmt} images (${R.near_dup_pct}) have a near-duplicate (pHash distance ≤ 4). ${R.cross_label_text} The copies are not spread evenly (${figRef("dup_class")}): ${R.dup.copy_share_range} of the cardboard, metal, paper and plastic files are redundant copies, glass has ${R.dup.glass_share} and trash none. ${R.dup_origin_text} Duplicates are dangerous because a random split would place copies of the same photo in both the training and the test set, so the test accuracy would measure memorisation rather than generalisation – a classic form of data leakage (Kaufman et al., 2012). This probably explains part of the very high scores reported by public notebooks that split this data set at random.`));
  out.push(...figure("dup_class", R.fig("fig01b_duplicates_by_class.png"), 14, "Unique Images and Byte-Identical Copies per Class",
    "A file is a redundant copy when another file of the data set has the same MD5 checksum; one file of each identical pair is counted as unique."));
  out.push(p("**Heterogeneous sources.** The images come from different collections: a large share has the plain white background of studio-style photos, while others are cluttered real-world scenes (Section 1.2). A model could learn shortcuts such as \"white background = source X = class Y\", which is why the evaluation also includes robustness and explainability checks."));
  if (R.trash_sources) {
    const ts = Object.entries(R.trash_sources).map(([k, v]) => `${k} (${Math.round(100 * v)}%)`);
    out.push(p(`**What the labels mean.** The class names do not always match everyday language or New Zealand's recycling rules. The provenance analysis shows that the *trash* images come from source folders named ${ts.slice(0, -1).join(", ")} and ${ts[ts.length - 1]} – i.e. mostly textiles, footwear, food waste and batteries rather than general rubbish – and drink cartons (liquid paperboard) are labelled *cardboard* although they are not accepted at kerbside (MfE, 2023). The model learns the labels as given, so the application adds rule-based advice for these cases (for example, batteries must go to a drop-off point).`));
  }
  out.push(p("**Pre-processing requirements** that follow from this analysis (applied in Section 2.2):"));
  out.push(...bullets([
    "remove exact duplicates and files whose identical copies carry different labels;",
    "keep near-duplicates together in one subset through a stratified *group* split, so that there is no leakage between training, validation and test data;",
    "resize/crop the 256 × 256 images to the 224 × 224 input of the network and normalise pixel values with the ImageNet statistics required by the pre-trained model;",
    "use data augmentation to compensate for the moderate data size and the variety of real photos (rotation, flips, crops, lighting);",
    "no class re-balancing is needed, and no missing values or corrupted files have to be handled.",
  ]));
  return out;
}

module.exports = { cover, frontMatter, section11 };

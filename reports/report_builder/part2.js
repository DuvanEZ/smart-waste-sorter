// Section 1.2 Data visualisation
const L = require("./lib");
const { p, h2, h3, bullets, figure, table, figRef, tabRef } = L;

const f2 = (x) => Number(x).toFixed(2);
const f3 = (x) => Number(x).toFixed(3);
const f0 = (x) => Math.round(Number(x)).toLocaleString("en-US");
const pc = (x, d = 0) => `${(100 * Number(x)).toFixed(d)}%`;

// class with the highest / lowest median of a feature
function extremes(med, feat) {
  const e = Object.entries(med[feat] || {}).sort((a, b) => b[1] - a[1]);
  return { hi: e[0], lo: e[e.length - 1], all: e };
}

function section12(R) {
  const out = [];
  const M = R.medians;
  const C = R.corr;
  const W = R.white_bg;           // share of images with a plain white border, per class
  const D = R.dup;                // duplicates per class

  out.push(h2("1.2 Data Visualisation"));
  out.push(h3("1.2.1 Variables and choice of techniques"));
  out.push(p(`An image data set is not a table: each 256 × 256 RGB image consists of 196,608 pixel values, which cannot be plotted one by one. The analysis therefore defined the **variables** that describe the data set and visualised every one of them, alone and in relation to the class:`));
  out.push(...bullets([
    "**Target variable** – the class label (categorical, six levels).",
    "**File variables** – width, height, file format, colour mode and file size.",
    "**Image-content variables** derived from the pixels with standard image-processing measures: brightness (mean grey level), contrast (standard deviation of the grey levels), saturation (mean HSV saturation), colourfulness (Hasler & Süsstrunk, 2003), sharpness (variance of the Laplacian; Pech-Pacheco et al., 2000), entropy (information content of the grey-level histogram), edge density (share of Canny edge pixels; Canny, 1986), share of near-white pixels on the image border (\"white background\"), the mean red, green and blue values, the full RGB and hue distributions, and the average image of each class.",
    "**Learned representation** – the 1,280 features that the pre-trained EfficientNet-B0 extracts from each image, i.e. the representation the model will use.",
    "**Relationships between images** – the perceptual-hash distance of every image to its most similar neighbour (duplicates) and the public source data set it matches (provenance).",
  ]));
  out.push(p(`${tabRef("viz_techniques")} lists each visualisation, the variables it shows, the type of technique and why it was chosen. Two principles guided the choices: the *type* of variable decides the chart (bars for categories, histograms and box plots for continuous variables, scatter plots and correlation matrices for relationships), and every chart that compares classes uses **the same colour for the same class** and small multiples, so the six classes can be compared without a legend look-up.`));
  out.push(...table("viz_techniques", "Visualisations Used, Variables Shown and Justification of the Technique", [
    "Figure", "Variable(s)", "Technique (type)", "Why this technique",
  ], [
    [figRef("class_dist"), "Class label", "Bar chart (univariate, categorical)", "Bar lengths on a common baseline give the most accurate comparison of category counts; percentages and a mean line show the balance directly"],
    [figRef("samples"), "Raw pixels", "Image grid (small multiples)", "Only direct inspection reveals content, backgrounds and labelling quality"],
    [figRef("dup_class"), "Class × exact-copy status", "Stacked bar chart (bivariate, categorical)", "Shows the part-to-whole split of every class into unique images and copies, i.e. the real class sizes"],
    [figRef("file_props"), "Width × height, format, colour mode, file size", "Scatter plot, bar charts, histogram (univariate)", "Scatter of width against height exposes any variation in resolution; bars suit categories; a histogram shows the shape (skew) of a continuous variable"],
    [figRef("boxplots"), "Eight numeric image statistics × class", "Box plots, small multiples (bivariate: numeric vs categorical)", "Compare median, spread and skew of many variables over six groups compactly and robustly to outliers"],
    [figRef("white_bg"), "Plain white background × class", "Bar chart (bivariate)", "Isolates a property that the model could use as a shortcut"],
    [figRef("rgb"), "R, G and B intensity distributions × class", "Line histograms, small multiples (distribution)", "Lines allow the three channels to be overlaid without hiding each other"],
    [figRef("hue"), "Hue of coloured pixels × class", "Heat map (class × 18 hue bins)", "Compares six 18-bin distributions at once; a colour-wheel strip makes the hue axis readable"],
    [figRef("mean_img"), "All pixels × class", "Average image (multivariate)", "Reveals the typical composition: object shape, position and background"],
    [figRef("corr"), "12 numeric variables", "Correlation heat map, Spearman ρ (multivariate, pairwise)", "Summarises 66 relationships in one view; Spearman's rank correlation is robust to the strongly skewed variables (sharpness, file size)"],
    [figRef("scatter"), "Brightness × saturation × class", "Scatter plots, one class highlighted per panel (bivariate)", "Shows the joint distribution and class overlap without over-plotting 14,000 points"],
    [figRef("tsne"), "1,280 deep features × class", "t-SNE embedding to 2-D (multivariate)", "Visualises high-dimensional similarity: are the classes separable in the representation used by the model?"],
    [figRef("dup_hist"), "Distance to the most similar image", "Histogram with threshold line (univariate)", "Separates copies from genuinely different photos and justifies the duplicate threshold"],
    [figRef("dup_examples"), "Pairs of matched images", "Image pairs", "Verifies that hash matches are real duplicates and inspects pairs with different labels"],
    [figRef("provenance"), "Source data set × class", "100% stacked bar chart (bivariate, categorical)", "Part-to-whole composition of each class by origin"],
  ], [10, 22, 25, 43], "Figures 1–3 appear in Section 1.1; the split and augmentation figures belong to Section 2.2.", { size: 16 }));

  // ---------------------------------------------------------------- target + file variables
  out.push(h3("1.2.2 Target variable and file properties"));
  out.push(p(`The **class label** is the only variable supplied with the data. ${figRef("class_dist")} shows that the nominal counts are nearly uniform (${R.share_min}–${R.share_max} of the images per class), but ${figRef("dup_class")} shows that this balance is partly artificial: once byte-identical copies are counted only once, the classes contain between ${f0(D.min_unique)} (${D.min_class}) and ${f0(D.max_unique)} (${D.max_class}) different photos, a ratio of ${f2(D.ratio_unique)} instead of ${R.imbalance_ratio}. The copies are concentrated in cardboard, metal, paper and plastic, where ${D.copy_share_range} of the files are redundant copies; glass has ${pc(D.share.glass)} and trash none. A plausible explanation is that the creator topped up the smaller classes with copies, or merged source collections that overlap, to reach the advertised class sizes.`));
  out.push(p(`${figRef("file_props")} visualises the four **file variables**. Width and height collapse to a single point (all ${R.n_images_fmt} images are 256 × 256), and the format and colour-mode bars contain one category each (JPEG, RGB), confirming that the images were standardised by the data set's creator and need no format conversion. The file-size histogram is right-skewed (median ${R.median_kb} KB, range ${R.size_range}); for images of identical dimensions the JPEG size measures how much detail an image contains, which is why it correlates strongly with edge density (ρ = ${f2(C.file_size_kb__edge_density)}) and sharpness (ρ = ${f2(C.file_size_kb__sharpness)}; ${figRef("corr")}). The small sizes also show strong JPEG compression, so fine textures are partly lost – one reason why the robustness tests in Section 2.4 include compression.`));
  out.push(...figure("file_props", R.fig("fig03_file_properties.png"), 16, "File Properties of All Images",
    "(a) Resolution (width × height, marker area proportional to the number of images); (b) file format; (c) colour mode; (d) distribution of the file size."));

  // ---------------------------------------------------------------- content by class
  out.push(h3("1.2.3 Image content by class: brightness, colour and texture"));
  const br = extremes(M, "brightness"), sat = extremes(M, "saturation"), col = extremes(M, "colourfulness");
  const sh = extremes(M, "sharpness"), ed = extremes(M, "edge_density"), en = extremes(M, "entropy");
  out.push(p(`${figRef("boxplots")} compares the distributions of eight numeric image variables across the classes. Each box shows the middle 50% of the images and the line the median. The main differences are:`));
  out.push(...bullets([
    `**Brightness and contrast.** Trash photos are the darkest (median brightness ${f3(M.brightness.trash)} vs ${f3(br.hi[1])} for ${br.hi[0]}) and have the highest contrast, because many show cluttered, real-world scenes rather than one item on a light background. Plastic has the lowest contrast (${f3(M.contrast.plastic)}): transparent and white plastics on white backgrounds produce flat images.`,
    `**Colour.** Metal is the least saturated and least colourful class (median saturation ${f3(M.saturation.metal)}, colourfulness ${f0(M.colourfulness.metal)}), as expected for grey aluminium and steel, whereas cardboard is the most saturated (${f3(M.saturation.cardboard)}) and the most colourful (${f0(M.colourfulness.cardboard)}) because of its brown colour and printed packaging; trash is also colourful (food, mixed packaging).`,
    `**Texture and detail.** Paper has the highest sharpness, entropy and edge density (median edge density ${f3(M.edge_density.paper)}, about twice that of glass, ${f3(M.edge_density.glass)}) because newspapers, magazines and crumpled sheets are full of text and fine structure; it also produces the largest files. Glass is the smoothest class (lowest ${ed.lo[0] === "glass" ? "edge density, " : ""}sharpness ${f0(M.sharpness.glass)} and entropy ${f2(M.entropy.glass)}): bottles and jars have large uniform surfaces.`,
    `**Overlap.** Despite these differences, the boxes of all classes overlap strongly for every variable: no single statistic separates the classes, so a classifier must combine many cues.`,
  ]));
  out.push(...figure("boxplots", R.fig("fig04_image_statistics_by_class.png"), 16.5, "Distribution of Eight Image Variables by Class",
    "Box plots: box = interquartile range (IQR), line = median, whiskers = 1.5 × IQR; outliers are hidden for readability. Sharpness is shown on a log10 scale because it spans several orders of magnitude."));
  out.push(p(`${figRef("white_bg")} isolates one variable that could act as a **shortcut**: the share of images whose border consists mostly of near-white pixels, i.e. objects photographed on a white sheet, as in TrashNet. It ranges from ${pc(W.paper, 1)} of paper images to ${pc(W.glass, 1)} of glass images. A model could partly learn "white background → glass", which would not hold for user photos; this is checked later with Grad-CAM (Section 2.4).`));
  out.push(...figure("white_bg", R.fig("fig04b_white_background.png"), 12.5, "Share of Images With a Plain White (Studio) Background per Class",
    "An image counts as \"white background\" when at least 60% of the pixels in its 10-pixel border are near-white (grey level > 200 and saturation < 40 on a 0–255 scale)."));
  out.push(p(`The **colour distributions** confirm these patterns in more detail (${figRef("rgb")}). Every class shows a peak at intensity 255 – pixels of white backgrounds and highlights – which is highest for glass (about 18% of all pixels) and smallest for paper, whose sheets usually fill the frame. Cardboard has red above green above blue across the mid-tones, the signature of brown; metal and plastic have almost identical R, G and B curves (neutral grey, silver and clear materials); glass has an extra peak of dark pixels from dark bottles and backgrounds; and trash has the flattest distributions, consistent with very varied scenes.`));
  out.push(...figure("rgb", R.fig("fig05_rgb_histograms.png"), 16, "Distribution of Red, Green and Blue Pixel Intensities per Class",
    "Average 32-bin histogram of each channel over all images of the class."));
  out.push(p(`The hue heat map (${figRef("hue")}) considers only clearly coloured pixels. Cardboard is dominated by orange-brown hues of 20–40° (more than a third of its coloured pixels), the clearest colour signature in the data set; paper and glass also show warm hues (paper from yellowish sheets and print, glass from brown bottles), plus some green for glass; plastic and metal have a noticeable blue component (200–220°) from blue bottles, caps and cool reflections; trash spreads over red hues (0–20° and 340–360°) from food and packaging. Colour is therefore informative but not decisive, which is why training uses no hue augmentation (Section 2.2).`));
  out.push(...figure("hue", R.fig("fig06_hue_heatmap.png"), 16, "Hue Distribution of the Coloured Pixels per Class",
    "Cell colour = percentage of the class's coloured pixels in each 20° hue bin; only pixels with saturation and value ≥ 40/255 are counted, so grey and white pixels are excluded. The strip under the map shows the colour of each hue."));
  out.push(p(`The **average image** of each class (${figRef("mean_img")}) summarises all pixels at once. The class means reveal typical compositions: a vertical bottle silhouette for glass (and faintly for plastic), a round grey object in the centre for metal (cans and lids), a brown centred box for cardboard, while paper and trash are uniform because sheets and scenes fill the whole frame. The light borders of most means again show the many plain backgrounds, and the centred objects justify centre-cropping in the pre-processing.`));
  out.push(...figure("mean_img", R.fig("fig07_mean_images.png"), 16, "Average Image of Each Class",
    "Pixel-wise mean of all images of the class."));

  // ---------------------------------------------------------------- relationships
  out.push(h3("1.2.4 Relationships between variables"));
  out.push(p(`${figRef("corr")} shows the Spearman rank correlations between the twelve numeric variables. Three groups of strongly related variables appear: (1) **brightness** with the mean red, green and blue values (ρ = ${f2(C.brightness__mean_r)}–${f2(C.brightness__mean_g)}) and with the white-background share (ρ = ${f2(C.brightness__white_background)}); (2) the **detail** variables sharpness, edge density and file size (ρ = ${f2(C.file_size_kb__sharpness)}–${f2(C.sharpness__edge_density)}); and (3) **colour**: saturation with colourfulness (ρ = ${f2(C.saturation__colourfulness)}). Brightness is negatively related to entropy (ρ = ${f2(C.brightness__entropy)}) and saturation (ρ = ${f2(C.brightness__saturation)}): bright images are mostly plain white backgrounds, which contain little information and little colour. Several variables are therefore redundant, but the groups are only weakly related to each other, so brightness, detail and colour are distinct dimensions of variation.`));
  out.push(...figure("corr", R.fig("fig08_correlation_heatmap.png"), 13.5, "Spearman Correlation Between the Numeric Image Variables",
    "Red = positive, blue = negative correlation; coefficients with |ρ| ≥ 0.5 are printed."));
  out.push(p(`${figRef("scatter")} plots the relationship between the two variables with the clearest class differences, brightness and saturation. All classes fill the same triangular region – very bright pixels cannot be strongly saturated, so saturation must fall as brightness approaches 1 – but they occupy different parts of it: cardboard lies higher (more saturated at the same brightness), metal lower, trash further left (darker), and glass contains a separate group of very dark images. The large overlap again shows that hand-crafted colour statistics cannot separate the six classes; this motivates a model that learns its own features.`));
  out.push(...figure("scatter", R.fig("fig09_brightness_vs_saturation.png"), 16, "Saturation Against Brightness, One Class Highlighted per Panel",
    "Random sample of 700 images per class; grey points show the images of all other classes."));
  out.push(p(`To see the data the way a deep network does, each image was passed through the ImageNet-pre-trained EfficientNet-B0 *without any training on waste images*, and the resulting 1,280-dimensional feature vectors were projected to two dimensions with t-SNE (van der Maaten & Hinton, 2008; ${figRef("tsne")}). Even without training, every class concentrates in its own regions: cardboard and paper form compact neighbouring clusters (they are both fibre-based and sometimes confused), glass and metal occupy the right and lower-right parts, trash forms several separate sub-clusters (consistent with a heterogeneous "everything else" class), and plastic is the most scattered, overlapping glass and metal. This visual evidence that pre-trained features already separate the classes supports the choice of **transfer learning** in Section 2.1, and the overlapping regions predict the confusions observed in Section 2.4.`));
  out.push(...figure("tsne", R.fig("fig12_tsne_embeddings.png"), 16, "t-SNE Map of Pre-Trained EfficientNet-B0 Features",
    "500 random images per class; each panel highlights one class over the others (grey). Distances between distant clusters in a t-SNE map are not meaningful; only neighbourhoods are."));

  // ---------------------------------------------------------------- duplicates + provenance
  out.push(h3("1.2.5 Duplicates and the origin of the images"));
  out.push(p(`For every image, the Hamming distance between its 64-bit perceptual hash and that of the most similar other image was computed (${figRef("dup_hist")}). The histogram is clearly **bimodal**: ${R.images_phash0_fmt} images have a twin with an identical hash (distance 0) and a few hundred more lie at distances 2–4, while the remaining images form a separate bell-shaped group around 12–16 bits. The gap between the two groups justifies the near-duplicate threshold of 4 bits. (Distances are always even because each pHash has exactly 32 bits set.) Exact checksums (MD5) confirm the picture: the data set contains ${R.exact_dup_groups_fmt} pairs of byte-identical files – every duplicate group is a pair – and ${R.cross_md5_groups} of these pairs carry two different labels.`));
  out.push(...figure("dup_hist", R.fig("fig10_near_duplicates.png"), 13, "Distance of Each Image to Its Most Similar Other Image",
    "Perceptual-hash (pHash) Hamming distance in bits; 0 = identical hashes. Orange bars: images treated as near-duplicates (distance ≤ 4)."));
  out.push(p(`${figRef("dup_examples")} shows example pairs. The same-class pairs at distance 0 are indeed the same photograph; the cross-class pairs at distance 4, however, are different objects (for example a milk carton and a glass bottle) whose hashes are similar only because both are small items on a white background. Grouping them is harmless – it only keeps them in the same subset – whereas the identical files with conflicting labels are genuine label errors and were removed (Section 2.2).`));
  out.push(...figure("dup_examples", R.fig("fig11_duplicate_examples.png"), 16, "Examples of Near-Duplicate Pairs",
    "Each column shows one pair (top and bottom image) with the two class labels and the pHash distance."));
  out.push(p(R.provenance_viz_text));
  out.push(...figure("provenance", R.fig("fig25_provenance.png"), 16, "Share of Each Class Matching Images of Public Source Data Sets",
    "An image matches a source when the pHash distance to one of the source's images is ≤ 6 bits (the source images were hashed after resizing and after centre-cropping, to undo the standardisation to 256 × 256). Images that match several sources are attributed to the closest match."));

  // ---------------------------------------------------------------- summary
  out.push(h3("1.2.6 Summary of insights and consequences for the model"));
  out.push(...bullets([
    "The images are technically uniform (256 × 256 RGB JPEG, no corrupted files), so pre-processing only needs resizing to the network input and normalisation.",
    `The apparent balance is inflated by copies: the real class sizes differ by a factor of ${f2(D.ratio_unique)}. This is moderate, so no re-sampling is used, but macro-averaged metrics are reported and used for model selection.`,
    `${R.exact_dup_pct} of the files belong to byte-identical pairs (half of them are redundant copies) and ${R.cross_md5_groups} pairs have conflicting labels: de-duplication and a group split are essential to avoid leakage.`,
    "Classes differ in colour (brown cardboard, grey metal), texture (detailed paper, smooth glass) and scene type (dark, cluttered trash), but every variable overlaps strongly between classes; only a model that learns combinations of many features can separate them – the t-SNE map shows that pre-trained CNN features already do so to a large extent.",
    "Backgrounds differ between classes and sources (white studio photos vs real scenes), a possible shortcut that the evaluation must check (Grad-CAM, robustness tests).",
    "Likely confusions are cardboard ↔ paper and plastic ↔ glass/metal, which guide the error analysis.",
  ]));
  return out;
}

module.exports = { section12 };

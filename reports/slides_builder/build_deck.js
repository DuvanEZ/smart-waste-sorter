// Presentation deck (8-10 minutes): node build_deck.js <report_data.json> <out.pptx>
// Uses the same data file as the report so that every number on the slides matches it.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");

const R = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const OUT = process.argv[3] || "presentation.pptx";
const PROJECT = path.resolve(__dirname, "..", "..");
const FIG = (n) => path.join(R.results_dir, "figures", n);
const APP = (n) => path.join(R.app_fig_dir, n);
const SAMPLE = (n) => path.join(PROJECT, "sample_images", n);

// ---- design tokens ----------------------------------------------------------------------
const C = {
  dark: "163B2A", green: "2E7D4F", mint: "9FD8B4", tint: "EEF6F0", tint2: "DCEFE2",
  amber: "E8912D", ink: "1E2A23", muted: "5F6B64", white: "FFFFFF", grey: "C9D1CC", red: "C0392B",
};
const CLASS = { cardboard: "2A78D6", glass: "EB6834", metal: "1BAF7A", paper: "EDA100", plastic: "E87BA4", trash: "008300" };
const CLASSES = Object.keys(CLASS);
const HEAD = "Cambria", BODY = "Calibri";
const W = 13.333, H = 7.5, MX = 0.6;

const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
const fmt = (n) => Math.round(n).toLocaleString("en-US");

// ---- helpers ----------------------------------------------------------------------------
async function icon(Comp, color = "#FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

async function imgData(file, maxW = 1800) {
  const buf = await sharp(file).resize({ width: maxW, withoutEnlargement: true }).png({ compressionLevel: 9 }).toBuffer();
  const meta = await sharp(buf).metadata();
  return { data: "image/png;base64," + buf.toString("base64"), w: meta.width, h: meta.height };
}

// place an image inside a box, keeping its aspect ratio (centred)
function placeImage(slide, img, x, y, w, h, extra = {}) {
  const r = img.w / img.h;
  let iw = w, ih = w / r;
  if (ih > h) { ih = h; iw = h * r; }
  slide.addImage({ data: img.data, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, ...extra });
  return { x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih };
}

function title(slide, text, opts = {}) {
  slide.addText(text, { x: MX, y: 0.38, w: W - 2 * MX, h: 0.8, fontFace: HEAD, fontSize: opts.size || 32, bold: true,
    color: opts.color || C.ink, margin: 0, valign: "middle", isTextBox: true });
}

function footer(slide, n, dark = false) {
  slide.addText(String(n), { x: W - 1.0, y: H - 0.45, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10,
    color: dark ? C.mint : C.muted, align: "right", margin: 0, isTextBox: true });
}

function card(slide, x, y, w, h, fill = C.tint) {
  slide.addShape("roundRect", { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.12 });
}

function iconBadge(slide, data, x, y, d = 0.5, fill = C.green) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  slide.addImage({ data, x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56 });
}

function bigStat(slide, x, y, w, value, label, color = C.green, valueSize = 40) {
  slide.addText(value, { x, y, w, h: 0.75, fontFace: HEAD, fontSize: valueSize, bold: true, color, margin: 0, isTextBox: true });
  slide.addText(label, { x, y: y + 0.75, w, h: 0.6, fontFace: BODY, fontSize: 13, color: C.muted, margin: 0, valign: "top", isTextBox: true });
}

function bulletList(slide, items, x, y, w, h, size = 15, color = C.ink) {
  const runs = [];
  items.forEach((it, i) => {
    const parts = Array.isArray(it) ? it : [{ text: it }];
    parts.forEach((pt, j) => runs.push({ text: pt.text, options: { bold: !!pt.bold, color: pt.color || color,
      bullet: j === 0 ? { indent: 15 } : undefined, breakLine: j === parts.length - 1 && i < items.length - 1 } }));
  });
  slide.addText(runs, { x, y, w, h, fontFace: BODY, fontSize: size, color, valign: "top", margin: 0,
    paraSpaceAfter: 8, isTextBox: true });
}

function chartBase(extra = {}) {
  return Object.assign({
    catAxisLabelColor: C.muted, valAxisLabelColor: C.muted, catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY,
    catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, valGridLine: { color: "E3E7E4", size: 0.75 },
    catGridLine: { style: "none" }, dataLabelFontFace: BODY, dataLabelFontSize: 10, dataLabelColor: C.ink,
    legendFontFace: BODY, legendFontSize: 11, legendColor: C.ink, titleFontFace: BODY,
  }, extra);
}

// ---- deck -------------------------------------------------------------------------------
async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.author = "Smart Waste Sorter project";
  pres.title = "Smart Waste Sorter – INFO813 project presentation";

  const ic = {
    recycle: await icon(fa.FaRecycle), camera: await icon(fa.FaCamera), brain: await icon(fa.FaBrain),
    bin: await icon(fa.FaTrashCan), copy: await icon(fa.FaCopy), db: await icon(fa.FaDatabase),
    palette: await icon(fa.FaPalette), layers: await icon(fa.FaLayerGroup), shield: await icon(fa.FaShieldHalved),
    warn: await icon(fa.FaTriangleExclamation), check: await icon(fa.FaCircleCheck), bulb: await icon(fa.FaLightbulb),
    arrow: await icon(fa.FaArrowRight, "#2E7D4F"), search: await icon(fa.FaMagnifyingGlass), image: await icon(fa.FaImage),
    sun: await icon(fa.FaSun), shapes: await icon(fa.FaShapes), gauge: await icon(fa.FaGaugeHigh),
    laptop: await icon(fa.FaLaptopCode), list: await icon(fa.FaListCheck), wrench: await icon(fa.FaWrench),
  };
  const T = R.test_metrics;
  const n = { s: 0 };
  const next = () => ++n.s;

  // ===== 1. Title ==========================================================================
  {
    const s = pres.addSlide(); s.background = { color: C.dark };
    s.addText("INFO813 Artificial Intelligence  ·  Project  ·  Trimester 3, 2026", { x: MX, y: 0.7, w: 7, h: 0.4,
      fontFace: BODY, fontSize: 14, color: C.mint, margin: 0, isTextBox: true });
    s.addText("Smart Waste Sorter", { x: MX, y: 1.5, w: 7.2, h: 1.2, fontFace: HEAD, fontSize: 54, bold: true, color: C.white, margin: 0, isTextBox: true });
    s.addText("Classifying recyclable waste from a photo with transfer learning – and telling you which bin to use",
      { x: MX, y: 2.75, w: 6.6, h: 1.2, fontFace: BODY, fontSize: 20, color: "D7EADF", margin: 0, valign: "top", isTextBox: true });
    s.addText([{ text: "[Your name]  ·  [Student ID]", options: { breakLine: true } }, { text: "Auckland Institute of Studies", options: {} }],
      { x: MX, y: 5.7, w: 6, h: 0.9, fontFace: BODY, fontSize: 16, color: C.white, margin: 0, isTextBox: true });
    const picks = ["cardboard_01.jpg", "glass_02.jpg", "metal_01.jpg", "paper_02.jpg", "plastic_02.jpg", "trash_01.jpg"];
    const x0 = 7.75, y0 = 0.85, cell = 1.65, gap = 0.18;
    for (let i = 0; i < 6; i++) {
      const cx = x0 + (i % 3) * (cell + gap), cy = y0 + Math.floor(i / 3) * (cell + 0.62);
      const img = await imgData(SAMPLE(picks[i]), 400);
      s.addImage({ data: img.data, x: cx, y: cy, w: cell, h: cell, rounding: true });
      const cls = picks[i].split("_")[0];
      s.addShape("roundRect", { x: cx + 0.2, y: cy + cell + 0.1, w: cell - 0.4, h: 0.32, fill: { color: CLASS[cls] }, line: { color: CLASS[cls] }, rectRadius: 0.08 });
      s.addText(cls, { x: cx + 0.2, y: cy + cell + 0.1, w: cell - 0.4, h: 0.32, fontFace: BODY, fontSize: 12, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    }
    s.addNotes(`(≈15 s) Hello, I'm [name]. Smart Waste Sorter recognises the material of one piece of waste from a photo – cardboard, glass, metal, paper, plastic or trash – and tells you which bin to use. About four minutes on the data, four on the model, then a two-minute demo.`);
    footer(s, next(), true);
  }

  // ===== 2. Problem ========================================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, "Which bin? A costly everyday decision");
    bigStat(s, MX, 1.55, 3.6, "2.01 bn t", "municipal solid waste generated worldwide in 2016 – 3.40 bn t expected by 2050 (Kaza et al., 2018)");
    bigStat(s, MX, 3.15, 3.6, "16%", "of what New Zealand households put in recycling bins cannot be recycled (Parker, 2023)", C.amber);
    bigStat(s, MX, 4.75, 3.6, "≈ 25%", "of Auckland's kerbside recycling is contaminated – about $3 M extra cost per year (Auckland Council, 2024)");
    // flow photo -> model -> material -> bin
    const steps = [["Photo", ic.camera], ["AI model", ic.brain], ["Material", ic.shapes], ["Right bin", ic.recycle]];
    const fx = 5.0, fy = 1.75;
    steps.forEach(([label, data], i) => {
      const x = fx + i * 2.0;
      iconBadge(s, data, x + 0.35, fy, 0.9, i === 1 ? C.amber : C.green);
      s.addText(label, { x: x - 0.1, y: fy + 1.0, w: 1.8, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: C.ink, align: "center", margin: 0, isTextBox: true });
      if (i < steps.length - 1) s.addImage({ data: ic.arrow, x: x + 1.45, y: fy + 0.3, w: 0.32, h: 0.32 });
    });
    card(s, 5.0, 3.55, 7.7, 2.55);
    s.addText("New Zealand standard kerbside recycling (since 1 Feb 2024)", { x: 5.3, y: 3.72, w: 7.2, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: C.dark, margin: 0, isTextBox: true });
    bulletList(s, ["Glass bottles and jars", "Paper and cardboard (no drink cartons)", "Plastic containers 1, 2 and 5", "Aluminium and steel tins and cans"],
      5.3, 4.2, 4.0, 1.7, 14);
    s.addText("Goal: recognise the material from ONE photo and turn it into disposal advice – a perception task that cannot be solved with hand-written rules.",
      { x: 9.3, y: 4.2, w: 3.2, h: 1.7, fontFace: BODY, fontSize: 13, italic: true, color: C.green, margin: 0, valign: "top", isTextBox: true });
    s.addText("Source: Ministry for the Environment (2023)", { x: 5.3, y: 5.75, w: 4, h: 0.3, fontFace: BODY, fontSize: 10, color: C.muted, margin: 0, isTextBox: true });
    s.addNotes(`(≈30 s) Why it matters: the world produced two billion tonnes of municipal waste in 2016. In New Zealand, 16% of what goes into recycling bins cannot be recycled, and contamination costs Auckland about three million dollars a year. Since 2024 every council accepts the same materials, so the remaining problem is recognising the material – a perception task with no explicit rules, which is where machine learning helps.`);
    footer(s, next());
  }

  // ===== 3. Data set =======================================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, `The data: ${R.n_images_fmt} photos in six classes`);
    const counts = CLASSES.map((c) => R.class_counts[c]);
    s.addChart(pres.charts.BAR, [{ name: "Images", labels: CLASSES, values: counts }], chartBase({
      x: MX, y: 1.35, w: 6.3, h: 4.6, barDir: "col", chartColors: [C.green], showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "#,##0", valAxisHidden: true, valGridLine: { style: "none" }, showLegend: false,
      showTitle: true, title: "Files per class", titleFontSize: 14, titleColor: C.ink, catAxisLabelFontSize: 12,
    }));
    s.addText(`Nominal balance: largest/smallest class = ${R.imbalance_ratio}`, { x: MX, y: 6.0, w: 6.3, h: 0.35, fontFace: BODY, fontSize: 12, color: C.muted, margin: 0, isTextBox: true });
    const facts = [
      [ic.db, "Source", "Kaggle – Garbage Images Dataset (2000/class), v5, MIT licence (Cofone, 2025)"],
      [ic.image, "Format", `256 × 256 px, RGB JPEG, ${R.total_mb} MB in total; no corrupted files`],
      [ic.list, "Labels", "One folder per class; metadata.csv agrees 100% with the folders"],
      [ic.search, "Variables analysed", "Class, file properties, brightness, contrast, colour, texture, background, deep features, duplicates"],
    ];
    facts.forEach(([data, head, txt], i) => {
      const y = 1.45 + i * 1.18;
      iconBadge(s, data, 7.35, y, 0.6);
      s.addText(head, { x: 8.15, y: y - 0.04, w: 4.6, h: 0.35, fontFace: BODY, fontSize: 15, bold: true, color: C.ink, margin: 0, isTextBox: true });
      s.addText(txt, { x: 8.15, y: y + 0.3, w: 4.6, h: 0.7, fontFace: BODY, fontSize: 13, color: C.muted, margin: 0, valign: "top", isTextBox: true });
    });
    s.addNotes(`(≈30 s) The data are Kaggle's Garbage Images Dataset, version 5: ${R.n_images_fmt} photos in six classes. Technically they are clean – every image is a 256-pixel RGB JPEG, nothing is corrupted and the metadata agrees with the folders – and the classes look balanced. Images have no columns, so I defined the variables myself: file properties, brightness, contrast, colour, texture, background, deep features and duplicates, and visualised each of them.`);
    footer(s, next());
  }

  // ===== 4. Provenance =====================================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, "Where do the images really come from?");
    if (R.provenance_share) {
      const share = R.provenance_share;    // {source: {class: share}}
      // same order and colours as the report figure (fig25): Mohamed, Kunwar, TrashNet, unidentified
      const colourOf = (k) => k.includes("12 classes") ? "2A78D6" : k.includes("Kunwar") ? "EB6834" : k.startsWith("TrashNet") ? "1BAF7A" : "7D8782";
      const rank = (k) => k.includes("12 classes") ? 0 : k.includes("Kunwar") ? 1 : k.startsWith("TrashNet") ? 2 : 3;
      const order = Object.keys(share).sort((a, b) => rank(a) - rank(b));
      const short = (k) => k.startsWith("TrashNet") ? "TrashNet (2016)" : k.includes("12 classes") ? "Garbage Classification 12 cl. (2021)" :
        k.includes("Kunwar") ? "Garbage Dataset (Kunwar)" : "not identified";
      const data = order.map((k) => ({ name: short(k), labels: CLASSES, values: CLASSES.map((c) => share[k][c]) }));
      s.addChart(pres.charts.BAR, data, chartBase({
        x: MX, y: 1.3, w: 7.4, h: 4.9, barDir: "bar", barGrouping: "percentStacked", catAxisOrientation: "maxMin",
        chartColors: order.map(colourOf),
        showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: '[<0.06]"";0%', dataLabelColor: C.white, dataLabelFontSize: 10,
        valAxisHidden: true, valGridLine: { style: "none" }, showLegend: true, legendPos: "b", catAxisLabelFontSize: 12,
      }));
      const P = R.provenance;
      const tn = Object.keys(share).find((k) => k.startsWith("TrashNet"));
      const trashSrc = Object.keys(share).filter((k) => k !== "unidentified").sort((a, b) => share[b].trash - share[a].trash)[0];
      card(s, 8.4, 1.4, 4.35, 4.75);
      bulletList(s, [
        [{ text: `${pct(P.identified / P.n, 0)} `, bold: true, color: C.green }, { text: "of the images matched three public data sets (perceptual hashing)" }],
        [{ text: "TrashNet ", bold: true }, { text: `(Stanford, 2016) supplies ${pct(Math.min(...["cardboard", "glass", "metal", "paper", "plastic"].map((c) => share[tn][c])), 0)}–${pct(Math.max(...["cardboard", "glass", "metal", "paper", "plastic"].map((c) => share[tn][c])), 0)} of each recyclable class` }],
        [{ text: `${pct(share[trashSrc].trash, 0)} of “trash” `, bold: true, color: C.amber }, { text: `comes from one collection${R.trash_sources ? " – mostly " + Object.keys(R.trash_sources).slice(0, 3).map((k) => (k === "biological" ? "food waste" : k)).join(", ") : ""}` }],
        [{ text: "Risk: ", bold: true, color: C.red }, { text: "classes differ in source style (background, camera), so a model could learn the source instead of the material" }],
      ], 8.65, 1.65, 3.9, 4.3, 14);
    } else {
      s.addText("Provenance results not available in this run.", { x: MX, y: 2, w: 8, h: 1, fontFace: BODY, fontSize: 18, color: C.muted, isTextBox: true });
    }
    s.addNotes(`(≈35 s) The brief asks where the data originally came from, and Kaggle only says "a big ensemble". So I compared every image with three public data sets using perceptual hashes, which survive resizing. ` + (R.provenance ? `${pct(R.provenance.identified / R.provenance.n, 0)} of the images could be traced. ` : "") + `The recyclable classes come largely from TrashNet – Stanford photos on a white poster board – while trash comes almost entirely from a different web collection. Different sources mean different photo styles: a shortcut risk I check later.`);
    footer(s, next());
  }

  // ===== 5. Duplicates =====================================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, `Hidden problem: ${Math.round(parseFloat(R.exact_dup_pct))}% of the files are identical pairs`);
    const D = R.dup;
    s.addChart(pres.charts.BAR, [
      { name: "unique photos", labels: CLASSES, values: CLASSES.map((c) => D.unique[c]) },
      { name: "byte-identical copies", labels: CLASSES, values: CLASSES.map((c) => D.copies[c]) },
    ], chartBase({
      x: MX, y: 1.3, w: 7.2, h: 4.9, barDir: "bar", barGrouping: "stacked", catAxisOrientation: "maxMin",
      chartColors: [C.green, C.amber], showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: "#,##0;;;",
      dataLabelColor: C.white, showLegend: true, legendPos: "b", valAxisLabelFormatCode: "#,##0", catAxisLabelFontSize: 12,
    }));
    bigStat(s, 8.3, 1.35, 4.5, `${R.exact_dup_files_fmt} files`, `(${R.exact_dup_pct}) form ${R.exact_dup_groups_fmt} identical pairs – none in trash, ${D.copy_share_range} redundant copies in four classes`, C.amber, 36);
    card(s, 8.3, 3.05, 4.45, 3.1);
    s.addText("Why it matters and what I did", { x: 8.55, y: 3.2, w: 4.0, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: C.dark, margin: 0, isTextBox: true });
    bulletList(s, [
      "Random split → copies in train AND test → inflated accuracy (data leakage)",
      `Real class sizes: ${D.min_unique_fmt}–${D.max_unique_fmt} (ratio ${D.ratio_unique.toFixed(2)}) → macro-F1 for model selection`,
      `Removed ${R.removed_exact_fmt} copies and ${R.removed_conflicts_fmt} files with conflicting labels`,
      "Near-duplicates (pHash ≤ 4) kept in the same subset: stratified group split",
    ], 8.55, 3.65, 4.0, 2.45, 13);
    s.addNotes(`(≈35 s) The key finding: although the data set claims to be de-duplicated, ${R.exact_dup_files_fmt} files – ${R.exact_dup_pct} – form byte-identical pairs, all in four classes. With a random split, copies would sit in both training and test sets and inflate the score. I removed the copies and the files with conflicting labels, and used a stratified group split so that near-duplicates never cross subsets. The real class sizes differ by ${R.dup.ratio_unique.toFixed(2)} times, so I select models by macro-F1.`);
    footer(s, next());
  }

  // ===== 6. Image statistics ===============================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, "What makes the classes different?");
    const img = await imgData(FIG("slides/slide_boxplots.png"), 2000);
    placeImage(s, img, MX, 1.25, 8.2, 5.1);
    const M = R.medians, Wb = R.white_bg;
    const cards = [
      [ic.palette, "Colour", `Cardboard is the most saturated (median ${M.saturation.cardboard.toFixed(2)}), metal the least (${M.saturation.metal.toFixed(2)}): brown vs grey`],
      [ic.layers, "Texture", `Paper has twice the edge density of glass (${M.edge_density.paper.toFixed(3)} vs ${M.edge_density.glass.toFixed(3)}): print and creases vs smooth surfaces`],
      [ic.sun, "Scene", `Trash photos are the darkest; ${pct(Wb.glass, 0)} of glass but ${pct(Wb.paper, 0)} of paper images have a white studio background`],
    ];
    cards.forEach(([data, head, txt], i) => {
      const y = 1.35 + i * 1.62;
      card(s, 9.05, y, 3.7, 1.45);
      iconBadge(s, data, 9.2, y + 0.15, 0.5);
      s.addText(head, { x: 9.85, y: y + 0.15, w: 2.8, h: 0.45, fontFace: BODY, fontSize: 15, bold: true, color: C.ink, margin: 0, valign: "middle", isTextBox: true });
      s.addText(txt, { x: 9.2, y: y + 0.68, w: 3.45, h: 0.72, fontFace: BODY, fontSize: 11.5, color: C.ink, margin: 0, valign: "top", isTextBox: true });
    });
    s.addText("But every variable overlaps between classes → no single rule separates them.", { x: MX, y: 6.4, w: 8.2, h: 0.4,
      fontFace: BODY, fontSize: 14, italic: true, color: C.green, margin: 0, isTextBox: true });
    s.addNotes(`(≈30 s) To compare classes I computed numeric variables per image and used box plots, which compare medians and spreads across six groups. Cardboard is the most saturated – brown boxes – and metal the least. Paper has the most edges and detail, glass the smoothest surfaces, and trash photos are darker, cluttered scenes. But every box overlaps: no single statistic separates the classes.`);
    footer(s, next());
  }

  // ===== 7. Relationships + t-SNE ==========================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, "Relationships between variables – and the network's view");
    const corr = await imgData(FIG("fig08_correlation_heatmap.png"), 1400);
    const tsne = await imgData(FIG("slides/slide_tsne.png"), 1800);
    placeImage(s, corr, MX, 1.3, 4.9, 4.6);
    placeImage(s, tsne, 5.75, 1.3, 7.0, 4.6);
    const Cr = R.corr;
    s.addText(`Spearman correlations: brightness ~ RGB (ρ up to ${Cr.brightness__mean_g.toFixed(2)}), sharpness ~ edges (${Cr.sharpness__edge_density.toFixed(2)}), saturation ~ colourfulness (${Cr.saturation__colourfulness.toFixed(2)})`,
      { x: MX, y: 6.0, w: 4.9, h: 0.8, fontFace: BODY, fontSize: 12, color: C.muted, margin: 0, valign: "top", isTextBox: true });
    s.addText("t-SNE of ImageNet features (no training yet): classes already form their own regions → transfer learning; overlaps predict confusions (paper–cardboard, plastic–glass/metal)",
      { x: 5.75, y: 6.0, w: 7.0, h: 0.8, fontFace: BODY, fontSize: 12, color: C.green, margin: 0, valign: "top", isTextBox: true });
    s.addNotes(`(≈35 s) The correlation map shows three groups of related variables: brightness, detail and colour. On the right, every image was passed through a network pre-trained on ImageNet – without any training on waste – and its 1,280 features were projected with t-SNE. The classes already form their own regions, which supports transfer learning, and the overlaps – paper with cardboard, plastic with glass and metal – predicted the errors I found later.`);
    footer(s, next());
  }

  // ===== 8. Model choice ===================================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, "Model: EfficientNet-B0 with transfer learning");
    const blocks = [
      ["Photo", "224 × 224 × 3"], ["Stem conv", "3 × 3, 32 filters"], ["16 MBConv blocks", "depthwise conv + squeeze-and-excitation"],
      ["Global avg. pooling", "1,280 features"], ["Dropout + dense", "6 outputs → soft-max"],
    ];
    blocks.forEach(([h, sub], i) => {
      const y = 1.35 + i * 0.98;
      const fill = i === 2 ? C.green : i === 4 ? C.amber : C.tint2;
      const txt = i === 2 || i === 4 ? C.white : C.ink;
      s.addShape("roundRect", { x: MX, y, w: 3.6, h: 0.78, fill: { color: fill }, line: { color: fill }, rectRadius: 0.1 });
      s.addText([{ text: h, options: { bold: true, breakLine: true } }, { text: sub, options: { fontSize: 11 } }],
        { x: MX + 0.15, y, w: 3.3, h: 0.78, fontFace: BODY, fontSize: 14, color: txt, margin: 0, valign: "middle", isTextBox: true });
    });
    s.addText("Pre-trained on 1.28 M ImageNet photos; only the last layer is new", { x: MX, y: 6.3, w: 3.7, h: 0.6, fontFace: BODY, fontSize: 11.5, italic: true, color: C.muted, margin: 0, isTextBox: true });
    // comparison chart
    const rows = R.baselineRows;
    const labels = rows.map((r) => r[0].replace(" + HOG/colour/LBP", " (hand-crafted)").replace("EfficientNet-B0 fine-tuned (final)", "EfficientNet-B0 fine-tuned").replace("Frozen EfficientNet-B0 + logistic regression", "Frozen EfficientNet + log. reg."));
    const acc = rows.map((r) => parseFloat(r[3]) / 100);
    const last = acc.length - 1;
    s.addChart(pres.charts.BAR, [
      { name: "alternatives", labels, values: acc.map((v, i) => (i === last ? 0 : v)) },
      { name: "chosen model", labels, values: acc.map((v, i) => (i === last ? v : 0)) },
    ], chartBase({
      x: 4.6, y: 1.25, w: 8.2, h: 4.4, barDir: "bar", barGrouping: "stacked", catAxisOrientation: "maxMin",
      chartColors: [C.grey, C.green], showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: "0.0%;;;",
      dataLabelColor: C.ink, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0%", showLegend: false,
      showTitle: true, title: "Test accuracy – all models trained on the same split", titleFontSize: 13, titleColor: C.ink, catAxisLabelFontSize: 11,
    }));
    bulletList(s, [
      [{ text: "Accurate: ", bold: true }, { text: "learned features beat hand-crafted ones by > 20 points" }],
      [{ text: "Data-efficient: ", bold: true }, { text: "≈ 8,000 training photos are enough" }],
      [{ text: "Light: ", bold: true }, { text: `${R.n_params_fmt} parameters, ${R.onnx_mb} MB, ≈ ${R.latency_ms} ms/image on a CPU` }],
    ], 4.75, 5.75, 8.0, 1.2, 13);
    s.addNotes(`(≈40 s) I compared five model families and chose transfer learning with EfficientNet-B0: a convolutional network whose depth, width and resolution are scaled together, pre-trained on 1.28 million ImageNet photos; only the last layer is replaced by six outputs. To justify the choice with evidence I trained the alternatives on the same split: classical models reach ${R.classical_range}, a CNN from scratch ${R.cnn_acc}, frozen features ${R.probe_acc}, and fine-tuning ${R.acc_pct}. It is also small and fast enough for a laptop without a GPU.`);
    footer(s, next());
  }

  // ===== 9. Pipeline =======================================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, "From raw files to a trained model");
    const sp = R.splitRows[R.splitRows.length - 1];
    const steps = [
      ["Clean", `remove ${R.removed_exact_fmt} copies + ${R.removed_conflicts_fmt} conflicts`],
      ["Group split", `${sp[1]} / ${sp[2]} / ${sp[3]} train / val / test`],
      ["Resize + normalise", "224 × 224, ImageNet mean/std"],
      ["Augment", "crop, flip, rotate, light, blur, noise"],
      ["Train 2 phases", `head ${R.cfg.head_epochs} ep → all layers ≤ ${R.cfg.finetune_epochs} ep`],
      ["Calibrate + export", `temperature ${R.temperature_2}, ONNX`],
    ];
    const bw = 1.92, gap = 0.12;
    steps.forEach(([h, sub], i) => {
      const x = MX + i * (bw + gap);
      card(s, x, 1.35, bw, 1.55, i === 4 ? C.tint2 : C.tint);
      s.addText(String(i + 1), { x: x + 0.12, y: 1.45, w: 0.45, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: C.amber, margin: 0, isTextBox: true });
      s.addText(h, { x: x + 0.12, y: 1.88, w: bw - 0.24, h: 0.38, fontFace: BODY, fontSize: 13.5, bold: true, color: C.ink, margin: 0, isTextBox: true });
      s.addText(sub, { x: x + 0.12, y: 2.24, w: bw - 0.24, h: 0.62, fontFace: BODY, fontSize: 11, color: C.muted, margin: 0, valign: "top", isTextBox: true });
    });
    const lc = await imgData(FIG("fig15_learning_curves.png"), 1800);
    placeImage(s, lc, MX, 3.15, 7.4, 3.75);
    card(s, 8.3, 3.2, 4.45, 3.0);
    s.addText("Training set-up", { x: 8.55, y: 3.32, w: 4, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: C.dark, margin: 0, isTextBox: true });
    bulletList(s, [
      `AdamW, lr ${R.cfg.head_lr} → ${R.cfg.finetune_lr}, warm-up + cosine decay`,
      "Cross-entropy with label smoothing 0.1",
      "Dropout 0.3, drop-path 0.1, weight decay 0.01",
      `Early stopping on validation macro-F1 (best epoch ${R.best_epoch})`,
      `Colab T4 GPU, ${R.train_time}`,
    ], 8.55, 3.8, 4.0, 3.0, 12.5);
    s.addNotes(`(≈35 s) The pipeline: clean, split 70/15/15 by groups, resize to 224 pixels and normalise as ImageNet expects. Training images are randomly cropped, flipped, rotated, re-lit, blurred and given noise – but the hue is not changed, because colour carries class information. Training has two phases: the new layer first, then the whole network with a smaller learning rate, label smoothing and early stopping. The learning curves fall together with a small gap: no over-fitting.`);
    footer(s, next());
  }

  // ===== 10. Iteration =====================================================================
  if (R.iteration) {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, "Iterating: what the first run taught me");
    const rows = R.iteration.rows;
    const pick = (label) => rows.find((r) => r[0].startsWith(label));
    const items = [
      [ic.warn, "Weak start", "Phase 1 (head only) under-performed: large random initial scores", "Zero-initialised new layer", pick("Best validation accuracy in phase 1")],
      [ic.camera, "Fragile to noisy photos", "Accuracy collapsed with sensor noise", "Blur + noise augmentation, no hue change", pick("Accuracy with Gaussian noise")],
      [ic.gauge, "Not converged", "Best epoch was the last one", "3 more fine-tuning epochs", pick("Test accuracy")],
    ];
    s.addText("Problem in run 1", { x: 1.45, y: 1.3, w: 4.2, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: C.muted, margin: 0, isTextBox: true });
    s.addText("Change", { x: 5.85, y: 1.3, w: 3.3, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: C.muted, margin: 0, isTextBox: true });
    s.addText("Run 1 → Run 2", { x: 9.4, y: 1.3, w: 3.3, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: C.muted, margin: 0, isTextBox: true });
    items.forEach(([data, head, prob, fix, row], i) => {
      const y = 1.8 + i * 1.6;
      card(s, MX, y, W - 2 * MX, 1.4, i % 2 ? C.white : C.tint);
      iconBadge(s, data, MX + 0.2, y + 0.4, 0.6, C.amber);
      s.addText([{ text: head, options: { bold: true, breakLine: true, fontSize: 15 } }, { text: prob, options: { fontSize: 12, color: C.muted } }],
        { x: 1.45, y: y + 0.15, w: 4.2, h: 1.1, fontFace: BODY, color: C.ink, margin: 0, valign: "middle", isTextBox: true });
      s.addImage({ data: ic.arrow, x: 5.6, y: y + 0.55, w: 0.25, h: 0.25 });
      s.addText(fix, { x: 5.95, y: y + 0.15, w: 3.2, h: 1.1, fontFace: BODY, fontSize: 13.5, bold: true, color: C.green, margin: 0, valign: "middle", isTextBox: true });
      if (row) s.addText([{ text: `${row[1]} → `, options: { color: C.muted } }, { text: row[2], options: { color: C.green, bold: true } }],
        { x: 9.4, y: y + 0.15, w: 3.3, h: 0.7, fontFace: HEAD, fontSize: 24, margin: 0, valign: "middle", isTextBox: true });
      if (row) s.addText(row[0], { x: 9.4, y: y + 0.85, w: 3.3, h: 0.45, fontFace: BODY, fontSize: 11, color: C.muted, margin: 0, isTextBox: true });
    });
    s.addNotes(`(≈30 s) Modelling was iterative. The first run showed three weaknesses: the head-only phase learned little because the new layer started with large random scores, accuracy collapsed on noisy photos, and the best epoch was the last one. A zero-initialised layer, blur and noise augmentation and three more epochs fixed them, as the numbers on the right show.`);
    footer(s, next());
  }

  // ===== 11. Results =======================================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, `Results on ${R.n_test_fmt} unseen test images`);
    bigStat(s, MX, 1.4, 3.3, R.acc_pct, `accuracy (95% CI ${R.acc_ci})`, C.green, 44);
    bigStat(s, 3.95, 1.4, 3.0, R.f1_3, "macro-F1 – all classes count equally", C.green, 44);
    bigStat(s, MX, 3.15, 3.3, R.top2_pct, "top-2 accuracy (right class among the two most likely)", C.amber, 36);
    bigStat(s, 3.95, 3.15, 3.0, R.auc_3, "macro ROC-AUC", C.amber, 36);
    const pcs = R.per_class.map((r) => r.f1);
    card(s, MX, 4.95, 6.35, 1.85);
    s.addText("F1 per class", { x: MX + 0.2, y: 5.05, w: 3, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: C.dark, margin: 0, isTextBox: true });
    R.per_class.forEach((r, i) => {
      const x = MX + 0.2 + i * 1.0;
      s.addShape("roundRect", { x, y: 5.5, w: 0.9, h: 0.3, fill: { color: CLASS[r.class] }, line: { color: CLASS[r.class] }, rectRadius: 0.06 });
      s.addText(r.class, { x, y: 5.5, w: 0.9, h: 0.3, fontFace: BODY, fontSize: 10, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(r.f1.toFixed(3), { x, y: 5.9, w: 0.9, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: C.ink, align: "center", margin: 0, isTextBox: true });
    });
    s.addText(`Errors: ${R.n_wrong} of ${R.n_test_fmt}`, { x: MX + 0.2, y: 6.4, w: 5.5, h: 0.3, fontFace: BODY, fontSize: 11, color: C.muted, margin: 0, isTextBox: true });
    const cm = await imgData(FIG("fig16_confusion_matrix.png"), 1400);
    placeImage(s, cm, 7.2, 1.25, 5.6, 5.6);
    s.addNotes(`(≈30 s) On ${R.n_test_fmt} test images never used for training or model selection, the model is right ${R.acc_pct} of the time – 95% interval ${R.acc_ci} – with macro-F1 ${R.f1_3}. Accuracy and macro-F1 are almost identical, so no class is neglected, and the correct class is among the top two in ${R.top2_pct} of cases.`);
    footer(s, next());
  }

  // ===== 12. Errors + Grad-CAM =============================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, "Understanding the errors – and what the model looks at");
    const conf = R.confused.slice(0, 4);
    s.addText("Most frequent confusions (test set)", { x: MX, y: 1.35, w: 6.2, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: C.dark, margin: 0, isTextBox: true });
    conf.forEach((c, i) => {
      const x = MX, y = 1.85 + i * 0.98;
      card(s, x, y, 6.3, 0.84);
      s.addShape("roundRect", { x: x + 0.18, y: y + 0.23, w: 1.25, h: 0.38, fill: { color: CLASS[c.true] }, line: { color: CLASS[c.true] }, rectRadius: 0.06 });
      s.addText(c.true, { x: x + 0.18, y: y + 0.23, w: 1.25, h: 0.38, fontFace: BODY, fontSize: 12, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addImage({ data: ic.arrow, x: x + 1.58, y: y + 0.3, w: 0.24, h: 0.24 });
      s.addShape("roundRect", { x: x + 1.97, y: y + 0.23, w: 1.25, h: 0.38, fill: { color: CLASS[c.predicted] }, line: { color: CLASS[c.predicted] }, rectRadius: 0.06 });
      s.addText(c.predicted, { x: x + 1.97, y: y + 0.23, w: 1.25, h: 0.38, fontFace: BODY, fontSize: 12, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(`${c.count} images (${pct(c.share_of_true_class)} of ${c.true})`, { x: x + 3.45, y: y + 0.17, w: 2.75, h: 0.5, fontFace: BODY, fontSize: 13, color: C.ink, margin: 0, valign: "middle", isTextBox: true });
    });
    s.addText([
      { text: "Similar-looking materials, not random mistakes. ", options: { bold: true } },
      { text: "The two most confident “errors” are a photo of corrugated board labelled paper – Grad-CAM shows the model looking at the corrugated edge, so the label is wrong, not the model." },
    ], { x: MX, y: 5.85, w: 6.3, h: 1.1, fontFace: BODY, fontSize: 13, color: C.green, margin: 0, valign: "top", isTextBox: true });
    const gc = await imgData(FIG("fig22_gradcam.png"), 1600);
    const box = placeImage(s, gc, 7.25, 1.2, W - 7.25 - MX, 5.65);
    s.addShape("rect", { x: box.x, y: box.y, w: box.w, h: box.h, fill: { type: "none" }, line: { color: C.grey, width: 0.75 } });
    s.addNotes(
      "(≈40 s) The errors are concentrated in pairs of similar materials – " +
      conf.slice(0, 3).map((c) => `${c.true} predicted as ${c.predicted}`).join(", ") +
      " – exactly the overlaps visible in the t-SNE map; some wrong images are even mislabelled in the data. " +
      (R.interp && R.interp.gradcam_short ? R.interp.gradcam_short : "Grad-CAM shows which pixels drove each decision, so I can check whether the model looks at the object or at the background."));
    footer(s, next());
  }

  // ===== 13. Calibration + robustness ======================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, "Can the user trust the confidence?");
    bigStat(s, MX, 1.4, 3.6, `${R.ece_before_3} → ${R.ece_3}`, `expected calibration error after temperature scaling (T = ${R.temperature_2})`, C.green, 34);
    bigStat(s, MX, 3.05, 3.6, `${R.acc_above_pct}`, `accuracy of the ${R.coverage_pct} of images accepted with the app's 60% threshold – the rest are flagged UNCERTAIN`, C.amber, 34);
    card(s, MX, 4.75, 3.7, 2.05);
    s.addText([{ text: "Robust to ", options: { bold: true } }, { text: "lighting changes, sensor noise and JPEG compression.", options: { breakLine: true } },
      { text: "Weakest: ", options: { bold: true } }, { text: R.robust_weak + " – the app asks for upright, well-lit colour photos." }],
      { x: MX + 0.2, y: 4.9, w: 3.35, h: 1.8, fontFace: BODY, fontSize: 13, color: C.ink, margin: 0, valign: "top", isTextBox: true });
    const rb = R.robust;
    s.addChart(pres.charts.BAR, [{ name: "accuracy", labels: rb.map((r) => r.corruption), values: rb.map((r) => r.accuracy) }], chartBase({
      x: 4.65, y: 1.25, w: 8.1, h: 5.6, barDir: "bar", catAxisOrientation: "maxMin", chartColors: [C.green],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0%", valAxisMinVal: 0, valAxisMaxVal: 1,
      valAxisLabelFormatCode: "0%", showLegend: false, showTitle: true, title: "Accuracy after common photo corruptions (test sample)",
      titleFontSize: 13, titleColor: C.ink, catAxisLabelFontSize: 12,
    }));
    s.addNotes(`(≈30 s) Because the app shows a confidence, it must be honest. One temperature fitted on the validation set reduced the calibration error from ${R.ece_before_3} to ${R.ece_3}. With the default 60% threshold, ${R.coverage_pct} of test images are accepted and ${R.acc_above_pct} of those are correct; the rest are flagged as uncertain. Lighting and compression barely matter; the weakest cases are ${R.robust_weak}.`);
    footer(s, next());
  }

  // ===== 14. Demo ==========================================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, "Live demo: the Smart Waste Sorter app");
    const shot = await imgData(APP("app_single.png"), 1800);
    const box = placeImage(s, shot, MX, 1.3, 7.6, 5.55);
    s.addShape("rect", { x: box.x, y: box.y, w: box.w, h: box.h, fill: { type: "none" }, line: { color: C.grey, width: 1 } });
    s.addText("Demo script (2 min)", { x: 8.55, y: 1.3, w: 4.2, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: C.dark, margin: 0, isTextBox: true });
    s.addText([
      { text: "Upload a photo → material, confidence, bin and tips", options: { bullet: { type: "number" }, breakLine: true } },
      { text: "Hard case → UNCERTAIN warning with second choice", options: { bullet: { type: "number" }, breakLine: true } },
      { text: "Batch: classify a folder with invalid files → explanations + CSV", options: { bullet: { type: "number" }, breakLine: true } },
      { text: "Model tab: test metrics and confusion matrix", options: { bullet: { type: "number" } } },
    ], { x: 8.55, y: 1.8, w: 4.2, h: 2.6, fontFace: BODY, fontSize: 13.5, color: C.ink, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true });
    card(s, 8.55, 4.55, 4.2, 2.3);
    bulletList(s, [
      [{ text: "Streamlit ", bold: true }, { text: "web app, runs locally; CLI alternative" }],
      [{ text: "ONNX Runtime: ", bold: true }, { text: `${R.onnx_mb} MB model, ≈ ${R.latency_ms} ms/image, no GPU` }],
      [{ text: "Validation: ", bold: true }, { text: "type, size, corruption, dimensions, folders" }],
      [{ text: "One-click start: ", bold: true }, { text: "run_app.bat / run_app.sh" }],
    ], 8.75, 4.75, 3.85, 2.0, 12.5);
    s.addNotes(
      "(≈2 min – switch to the browser) 1) Upload a photo of a plastic bottle: the app shows PLASTIC, the calibrated confidence, the recycling bin and tips, " +
      "and the probability of every class. 2) Upload the hard example of soup cartons: confidence is below 60%, so it says UNCERTAIN and gives the second choice. " +
      "3) In Batch classification, type the path of the demo folder containing a damaged file, a tiny icon and a panorama: valid images are classified, invalid ones " +
      "are rejected with a clear reason, and I can download the CSV. 4) The Model performance tab shows the test metrics. The app needs only five libraries and " +
      "starts with one double-click on run_app.bat. (Backup if the live demo fails: use these screenshots.)");
    footer(s, next());
  }

  // ===== 15. Conclusion ====================================================================
  {
    const s = pres.addSlide(); s.background = { color: C.dark };
    title(s, "Conclusion", { color: C.white, size: 36 });
    const cols = [
      [ic.check, "What worked", [
        `Critical data analysis: provenance traced, ${R.exact_dup_pct} duplicate files found and handled`,
        `EfficientNet-B0 transfer learning: ${R.acc_pct} accuracy, macro-F1 ${R.f1_3}`,
        "Calibrated confidence and an app that validates every input"]],
      [ic.warn, "Limitations", [
        "Only six classes: unknown objects are forced into one of them",
        "Classes come from different sources (style shortcut risk)",
        "Plastic resin codes (1, 2, 5) cannot be read from a photo"]],
      [ic.bulb, "Future work", [
        "Local New Zealand photos and an 'other' class",
        "Object detection for several items per photo",
        "Smaller quantised model for phones and smart bins"]],
    ];
    cols.forEach(([data, head, items], i) => {
      const x = MX + i * 4.1;
      s.addShape("roundRect", { x, y: 1.5, w: 3.85, h: 3.95, fill: { color: "1F4D37" }, line: { color: "1F4D37" }, rectRadius: 0.12 });
      iconBadge(s, data, x + 0.25, 1.75, 0.65, i === 1 ? C.amber : C.green);
      s.addText(head, { x: x + 1.05, y: 1.8, w: 2.6, h: 0.55, fontFace: HEAD, fontSize: 20, bold: true, color: C.white, margin: 0, valign: "middle", isTextBox: true });
      bulletList(s, items, x + 0.3, 2.65, 3.3, 2.7, 14, "E6F2EA");
    });
    s.addText("Thank you – questions?", { x: MX, y: 5.95, w: 8, h: 0.6, fontFace: HEAD, fontSize: 24, italic: true, color: C.mint, margin: 0, isTextBox: true });
    s.addNotes(`(≈20 s) To conclude: a critical data analysis found the duplicates that would have inflated the results; transfer learning reached ${R.acc_pct} accuracy with honest confidence, inside an application that validates its inputs. The main limitations are the six fixed classes and the mixed sources. Thank you – I'm happy to take questions.`);
    footer(s, next(), true);
  }

  // ===== 16. References ====================================================================
  {
    const s = pres.addSlide(); s.background = { color: C.white };
    title(s, "References", { size: 28 });
    const refs = [
      "Anthropic. (2026). Claude (Opus 5.5 version) [Large language model]. https://claude.ai",
      "Auckland Council. (2024, January). New standards to recycle right. OurAuckland.",
      "Cofone, L. (2025). Garbage images dataset (2000/class) (Version 5) [Data set]. Kaggle.",
      "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML, 1321–1330.",
      "Kaza, S., Yao, L. C., Bhada-Tata, P., & Van Woerden, F. (2018). What a waste 2.0. World Bank.",
      "Kunwar, S. (2026). The Garbage Dataset (GD): A multi-class image benchmark for automated waste segregation (arXiv:2602.10500).",
      "Ministry for the Environment. (2023). Standard materials for kerbside collections. New Zealand Government.",
      "Mohamed, M. (2021). Garbage classification (12 classes) [Data set]. Kaggle.",
      "Parker, D. (2023, March 29). Standard kerbside recycling part of new era for waste system [Press release].",
      "Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV, 618–626.",
      "Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML, 6105–6114.",
      "Thung, G., & Yang, M. (2016). Classification of trash for recyclability status [CS229 project report]. Stanford University.",
      "van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9, 2579–2605.",
      "Zauner, C. (2010). Implementation and benchmarking of perceptual image hash functions [Master's thesis].",
    ];
    const half = Math.ceil(refs.length / 2);
    [refs.slice(0, half), refs.slice(half)].forEach((col, k) => {
      s.addText(col.map((r, i) => ({ text: r, options: { breakLine: i < col.length - 1, indentLevel: 0 } })),
        { x: MX + k * 6.2, y: 1.3, w: 5.9, h: 5.6, fontFace: BODY, fontSize: 11, color: C.ink, margin: 0, valign: "top", paraSpaceAfter: 7, isTextBox: true });
    });
    s.addNotes("Full reference list in APA 7 format is in the written report.");
    footer(s, next());
  }

  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });

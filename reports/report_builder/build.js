// Assemble the report: node build.js <report_data.json> <output.docx>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, Header, Footer, PageNumber, HeadingLevel,
} = require("docx");
const L = require("./lib");
const part1 = require("./part1");
const part2 = require("./part2");
const part3 = require("./part3");
const part4 = require("./part4");
const part5 = require("./part5");

const dataFile = process.argv[2];
const outFile = process.argv[3] || "report.docx";
const R = JSON.parse(fs.readFileSync(dataFile, "utf8"));
// Figures are read from the down-scaled cache made by prepare_report_data.py (photo-like ones as .jpg)
R.fig = (name) => {
  const dir = R.fig_dir || path.join(R.results_dir, "figures");
  const jpg = path.join(dir, name.replace(/\.png$/, ".jpg"));
  return fs.existsSync(jpg) ? jpg : path.join(dir, name);
};
R.figStatic = (name) => path.join(R.static_fig_dir, name);
R.figApp = (name) => path.join(R.app_fig_dir, name);

// Figure / table numbering in order of appearance
L.registerOrder(
  ["class_dist", "samples", "dup_class", "file_props", "boxplots", "white_bg", "rgb", "hue", "mean_img", "corr", "scatter",
   "tsne", "dup_hist", "dup_examples", "provenance",
   "pipeline", "split", "augmentation", "learning",
   "comparison", "cm", "per_class", "roc", "reliability", "confidence", "misclassified", "gradcam", "robustness",
   "app_single", "app_uncertain", "app_batch", "app_invalid"],
  ["dataset_summary", "viz_techniques", "model_options", "baselines", "split", "hyper", "iteration", "eval_methods",
   "test_metrics", "per_class", "confused", "modules", "validation", "guide_tabs", "outputs", "trouble", "libraries"]);

const children = [
  ...part1.frontMatter(R),
  ...part1.section11(R),
  ...part2.section12(R),
  ...part3.section21(R),
  ...part3.section22(R),
  ...part3.section23(R),
  ...part4.section24(R),
  ...part5.section31(R),
  ...part5.section32(R),
  ...part5.conclusion(R),
  ...part5.references(),
  ...part5.appendices(R),
];

const headingStyle = (id, name, size, color, before, after, italics = false) => ({
  id, name, basedOn: "Normal", next: "Normal", quickFormat: true,
  run: { size, bold: true, color, font: L.FONT, italics },
  paragraph: { spacing: { before, after }, keepNext: true, keepLines: true,
               outlineLevel: { Heading1: 0, Heading2: 1, Heading3: 2 }[id] },
});

const doc = new Document({
  creator: "Smart Waste Sorter project",
  title: "Smart Waste Sorter – INFO813 Project Report",
  description: "Classifying recyclable waste images with transfer learning",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: L.FONT, size: 22, color: L.INK } } },
    paragraphStyles: [
      headingStyle("Heading1", "Heading 1", 32, L.GREEN, 360, 160),
      headingStyle("Heading2", "Heading 2", 27, L.GREEN, 300, 120),
      headingStyle("Heading3", "Heading 3", 23, "2F2F2F", 220, 100),
    ],
  },
  numbering: L.numberingConfig(),
  sections: [
    {
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      children: part1.cover(R),
    },
    {
      // page numbers continue from the cover (the cover itself shows no number), so that the
      // numbers in the table of contents and in the footers agree in Word and LibreOffice
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "INFO813 Project – Smart Waste Sorter", size: 17, color: L.MUTED })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 17, color: L.MUTED })] })] }) },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outFile, buf);
  console.log("wrote", outFile, (buf.length / 1e6).toFixed(1), "MB");
});

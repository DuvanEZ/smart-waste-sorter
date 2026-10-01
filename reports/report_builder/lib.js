// Helpers for building the project report with docx-js (APA-style figures and tables).
const fs = require("fs");
const path = require("path");
const {
  Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle, AlignmentType,
  HeadingLevel, ShadingType, PageBreak, VerticalAlign, TableLayoutType, LevelFormat,
} = require("docx");

const FONT = "Calibri";
const MONO = "Consolas";
const INK = "1A1A1A";
const GREEN = "1E5B3A";      // headings (waste / recycling theme, dark green)
const MUTED = "5A5A5A";
const CONTENT_WIDTH = 9026;  // A4 width 11906 - 2 x 1440 margins (DXA)

// ---------------------------------------------------------------- inline markup
// **bold**, *italic*, `code` inside a string -> TextRuns
function runs(text, base = {}) {
  text = fillRefs(String(text));
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...base }));
    else if (tok.startsWith("`")) out.push(new TextRun({ text: tok.slice(1, -1), font: MONO, size: 19, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), italics: true, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

function p(text, opts = {}) {
  return new Paragraph({
    children: Array.isArray(text) ? text : runs(text, opts.run || {}),
    alignment: opts.align || AlignmentType.LEFT,
    spacing: { after: opts.after ?? 140, before: opts.before ?? 0, line: opts.line ?? 288 },
    indent: opts.indent,
    keepNext: opts.keepNext,
    style: opts.style,
  });
}

const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: false });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

function bullets(items, level = 0) {
  return items.map((it) => new Paragraph({
    numbering: { reference: "bullets", level },
    children: runs(it),
    spacing: { after: 70, line: 276 },
  }));
}

let numberedCounter = 0;
function numbered(items) {
  numberedCounter += 1;
  const ref = `numbered-${numberedCounter}`;
  NUMBERING_REFS.push(ref);
  return items.map((it) => new Paragraph({
    numbering: { reference: ref, level: 0 },
    children: runs(it),
    spacing: { after: 70, line: 276 },
  }));
}
const NUMBERING_REFS = [];

function numberingConfig() {
  return {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1080, hanging: 270 } } } },
        ],
      },
      ...NUMBERING_REFS.map((ref) => ({
        reference: ref,
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
                   style: { paragraph: { indent: { left: 540, hanging: 300 } } } }],
      })),
    ],
  };
}

// ---------------------------------------------------------------- figures & tables (APA 7 style)
const FIG = {}; const TAB = {};
function registerOrder(figKeys, tabKeys) {
  figKeys.forEach((k, i) => { FIG[k] = i + 1; });
  tabKeys.forEach((k, i) => { TAB[k] = i + 1; });
}
const figRef = (k) => { if (!(k in FIG)) throw new Error("unknown figure " + k); return `Figure ${FIG[k]}`; };
const tabRef = (k) => { if (!(k in TAB)) throw new Error("unknown table " + k); return `Table ${TAB[k]}`; };
// Texts prepared in Python cannot call figRef/tabRef, so they contain the literal
// placeholders ${figRef("key")} / ${tabRef("key")}, which are resolved here.
function fillRefs(text) {
  return text.replace(/\$\{(figRef|tabRef)\("([^"]+)"\)\}/g, (_, fn, key) => (fn === "figRef" ? figRef(key) : tabRef(key)));
}

function pngSize(file) {
  const buf = fs.readFileSync(file);
  if (buf.toString("ascii", 1, 4) === "PNG") return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20), buf, type: "png" };
  // JPEG: scan for SOF marker
  let i = 2;
  while (i < buf.length) {
    if (buf[i] !== 0xff) { i++; continue; }
    const marker = buf[i + 1];
    const len = buf.readUInt16BE(i + 2);
    if (marker >= 0xc0 && marker <= 0xc3) return { h: buf.readUInt16BE(i + 5), w: buf.readUInt16BE(i + 7), buf, type: "jpg" };
    i += 2 + len;
  }
  throw new Error("unknown image size " + file);
}

function image(file, widthCm) {
  const { w, h, buf, type } = pngSize(file);
  const widthPx = Math.round(widthCm / 2.54 * 96);
  const heightPx = Math.round(widthPx * h / w);
  return new ImageRun({ data: buf, type, transformation: { width: widthPx, height: heightPx },
                        altText: { title: path.basename(file), description: path.basename(file), name: path.basename(file) } });
}

function figure(key, file, widthCm, title, note) {
  if (!fs.existsSync(file)) {
    console.warn("missing figure file:", file);
    return [new Paragraph({ children: [new TextRun({ text: `${figRef(key)} – ${title} [figure file missing: ${path.basename(file)}]`, bold: true, color: "B00020" })] })];
  }
  const out = [
    new Paragraph({ children: [new TextRun({ text: figRef(key), bold: true })], spacing: { before: 200, after: 40 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: title, italics: true })], spacing: { after: 80 }, keepNext: true }),
    new Paragraph({ children: [image(file, widthCm)], alignment: AlignmentType.CENTER, spacing: { after: 80 }, keepNext: !!note }),
  ];
  if (note) out.push(new Paragraph({ children: [new TextRun({ text: "Note. ", italics: true, size: 19 }), ...runs(note, { size: 19 })],
                                     spacing: { after: 220, line: 264 } }));
  return out;
}

function cell(text, { bold = false, align = AlignmentType.LEFT, width, top, bottom, shade, size = 18 } = {}) {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const line = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: "404040" });
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders: { top: top ? line(top) : none, bottom: bottom ? line(bottom) : none, left: none, right: none },
    shading: shade ? { fill: shade, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: align, spacing: { after: 0, line: 252 },
                               children: runs(String(text), { bold, size }) })],
  });
}

// APA table: horizontal rules only (top, below header, bottom)
function table(key, title, headers, rows, widthsPct, note, opts = {}) {
  const total = CONTENT_WIDTH;
  const widths = widthsPct.map((w) => Math.round(total * w / 100));
  widths[widths.length - 1] = total - widths.slice(0, -1).reduce((a, b) => a + b, 0);
  const align = opts.align || headers.map((_, i) => (i === 0 ? AlignmentType.LEFT : AlignmentType.LEFT));
  const size = opts.size || 18;
  const hdr = new TableRow({ tableHeader: true, children: headers.map((h, i) =>
    cell(h, { bold: true, width: widths[i], top: 10, bottom: 6, align: align[i], size })) });
  const body = rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) =>
    cell(c, { width: widths[i], bottom: ri === rows.length - 1 ? 10 : 0, align: align[i], size,
              shade: opts.highlightRow === ri ? "EAF3EC" : undefined,
              bold: opts.boldRow === ri })) }));
  const out = [
    new Paragraph({ children: [new TextRun({ text: tabRef(key), bold: true })], spacing: { before: 200, after: 40 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: title, italics: true })], spacing: { after: 80 }, keepNext: true }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
                rows: [hdr, ...body] }),
  ];
  if (note) out.push(new Paragraph({ children: [new TextRun({ text: "Note. ", italics: true, size: 19 }), ...runs(note, { size: 19 })],
                                     spacing: { before: 60, after: 220, line: 264 } }));
  else out.push(new Paragraph({ children: [], spacing: { after: 160 } }));
  return out;
}

function code(lines, opts = {}) {
  return lines.map((ln, i) => new Paragraph({
    children: [new TextRun({ text: ln.length ? ln : " ", font: MONO, size: opts.size || 17, color: "1F2937" })],
    shading: { fill: "F3F4F6", type: ShadingType.CLEAR, color: "auto" },
    spacing: { after: 0, before: i === 0 ? 60 : 0, line: 240 },
    indent: { left: 200, right: 200 },
    keepLines: true, keepNext: i < lines.length - 1,
  })).concat([new Paragraph({ children: [], spacing: { after: 120 } })]);
}

function reference(text) {
  // APA reference list entry with hanging indent; *italic* supported
  return new Paragraph({ children: runs(text), indent: { left: 720, hanging: 720 }, spacing: { after: 120, line: 264 } });
}

module.exports = { FONT, MONO, INK, GREEN, MUTED, CONTENT_WIDTH, runs, p, h1, h2, h3, pageBreak, bullets, numbered,
  numberingConfig, registerOrder, figRef, tabRef, fillRefs, figure, table, code, reference, image, FIG, TAB };

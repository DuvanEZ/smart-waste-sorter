"""Embed files (the appendix zip archives) into the report as OLE objects shown as icons,
exactly like Word's Insert > Object > Create from file > Display as icon.

    python embed_files.py report.docx out.docx file1.zip [file2.zip ...]

Each file replaces the paragraph that contains the text "[EMBEDDED FILE: <file name>]".
Double-clicking the icon in Word opens (or saves) the embedded zip archive.
"""
from __future__ import annotations

import io
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OLE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/oleObject"
IMG_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
SHAPETYPE = (
    '<v:shapetype id="_x0000_t75" coordsize="21600,21600" o:spt="75" o:preferrelative="t" '
    'path="m@4@5l@4@11@9@11@9@5xe" filled="f" stroked="f"><v:stroke joinstyle="miter"/><v:formulas>'
    '<v:f eqn="if lineDrawn pixelLineWidth 0"/><v:f eqn="sum @0 1 0"/><v:f eqn="sum 0 0 @1"/>'
    '<v:f eqn="prod @2 1 2"/><v:f eqn="prod @3 21600 pixelWidth"/><v:f eqn="prod @3 21600 pixelHeight"/>'
    '<v:f eqn="sum @0 0 1"/><v:f eqn="prod @6 1 2"/><v:f eqn="prod @7 21600 pixelWidth"/>'
    '<v:f eqn="sum @8 21600 0"/><v:f eqn="prod @7 21600 pixelHeight"/><v:f eqn="sum @10 21600 0"/>'
    '</v:formulas><v:path o:extrusionok="f" gradientshapeok="t" o:connecttype="rect"/>'
    '<o:lock v:ext="edit" aspectratio="t"/></v:shapetype>')


def font(size, bold=False):
    for name in (["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf"]):
        for folder in ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation"]:
            p = Path(folder) / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def icon_png(label: str, size_mb: float) -> bytes:
    """A zip-archive icon with the file name underneath (what Word displays)."""
    W, H = 820, 300
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    # folder-like archive symbol
    x0, y0 = W // 2 - 70, 18
    d.rounded_rectangle([x0, y0, x0 + 140, y0 + 170], radius=14, fill="#1E5B3A")
    d.rectangle([x0 + 62, y0, x0 + 78, y0 + 120], fill="#9FD8B4")
    for k in range(6):
        d.rectangle([x0 + 62 + (k % 2) * 8, y0 + 8 + k * 18, x0 + 70 + (k % 2) * 8, y0 + 18 + k * 18], fill="#1E5B3A")
    d.rounded_rectangle([x0 + 52, y0 + 118, x0 + 88, y0 + 150], radius=6, fill="#E8912D")
    d.text((x0 + 70, y0 + 160), "ZIP", fill="white", font=font(30, True), anchor="mb")
    size = 30
    while size > 16 and d.textlength(label, font=font(size, True)) > W - 30:
        size -= 1
    d.text((W // 2, 222), label, fill="#1A1A1A", font=font(size, True), anchor="mm")
    d.text((W // 2, 268), f"{size_mb:.1f} MB – double-click to open", fill="#5A5A5A", font=font(24), anchor="mm")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def main(src: str, dst: str, files: list[str]) -> None:
    with zipfile.ZipFile(src) as zin:
        parts = {n: zin.read(n) for n in zin.namelist()}
    doc = parts["word/document.xml"].decode("utf-8")
    rels = parts["word/_rels/document.xml.rels"].decode("utf-8")
    types = parts["[Content_Types].xml"].decode("utf-8")
    shapetype_done = False
    for k, f in enumerate(files, start=1):
        f = Path(f)
        marker = f"[EMBEDDED FILE: {f.name}]"
        # the whole paragraph that contains the marker
        m = re.search(r"<w:p>(?:(?!<w:p>).)*?" + re.escape(marker) + r".*?</w:p>", doc, flags=re.S)
        if not m:
            m = re.search(r"<w:p [^>]*>(?:(?!<w:p[ >]).)*?" + re.escape(marker) + r".*?</w:p>", doc, flags=re.S)
        if not m:
            raise SystemExit(f"placeholder not found: {marker}")
        with tempfile.TemporaryDirectory() as tmp:
            ole_path = Path(tmp) / "ole.bin"
            subprocess.run(["node", str(HERE / "make_ole.js"), str(f), str(ole_path)], check=True)
            ole = ole_path.read_bytes()
        ole_name, img_name = f"embeddings/oleObject{k}.bin", f"media/ole_icon{k}.png"
        parts[f"word/{ole_name}"] = ole
        parts[f"word/{img_name}"] = icon_png(f.name, f.stat().st_size / 1e6)
        rid_ole, rid_img = f"rIdOle{k}", f"rIdOleIcon{k}"
        rels = rels.replace("</Relationships>",
                            f'<Relationship Id="{rid_ole}" Type="{OLE_REL}" Target="{ole_name}"/>'
                            f'<Relationship Id="{rid_img}" Type="{IMG_REL}" Target="{img_name}"/></Relationships>')
        shape_id = f"_x0000_i10{24 + k}"
        obj = ('<w:p><w:pPr><w:spacing w:before="120" w:after="200"/><w:jc w:val="left"/></w:pPr><w:r>'
               '<w:object w:dxaOrig="4100" w:dyaOrig="1500">'
               + ("" if shapetype_done else SHAPETYPE) +
               f'<v:shape id="{shape_id}" type="#_x0000_t75" style="width:205pt;height:75pt" o:ole="">'
               f'<v:imagedata r:id="{rid_img}" o:title=""/></v:shape>'
               f'<o:OLEObject Type="Embed" ProgID="Package" ShapeID="{shape_id}" DrawAspect="Icon" '
               f'ObjectID="_17908{k:05d}" r:id="{rid_ole}"/></w:object></w:r></w:p>')
        shapetype_done = True
        doc = doc[:m.start()] + obj + doc[m.end():]
    if 'Extension="bin"' not in types:
        types = types.replace("<Default ", '<Default Extension="bin" ContentType="application/vnd.openxmlformats-officedocument.oleObject"/><Default ', 1)
    if 'Extension="png"' not in types:
        types = types.replace("<Default ", '<Default Extension="png" ContentType="image/png"/><Default ', 1)
    parts["word/document.xml"] = doc.encode("utf-8")
    parts["word/_rels/document.xml.rels"] = rels.encode("utf-8")
    parts["[Content_Types].xml"] = types.encode("utf-8")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        # [Content_Types].xml first, as Office writes it
        zout.writestr("[Content_Types].xml", parts.pop("[Content_Types].xml"))
        for name, data in parts.items():
            zout.writestr(name, data)
    print("wrote", dst, f"{Path(dst).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])

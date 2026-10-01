// Wrap a file (e.g. a .zip) in an OLE "Package" object, the format Word uses for
// Insert > Object > Create from file. Usage: node make_ole.js <file> <out.bin>
// The result is placed in word/embeddings/ of a .docx by embed_files.py.
const fs = require("fs");
const path = require("path");
// SheetJS cfb always adds a placeholder stream ("\u0001Sh33tJ5") to new containers; load a copy of
// the library with that seeding disabled so the OLE object contains only the standard streams.
const Module = require("module");
const cfbSrc = fs.readFileSync(require.resolve("cfb/cfb.js"), "utf8")
  .replace("function seed_cfb(cfb) {", "function seed_cfb(cfb) { return;");
const cfbModule = new Module("cfb-noseed", module);
cfbModule.paths = module.paths;
cfbModule._compile(cfbSrc, path.join(__dirname, "cfb-noseed.js"));
const CFB = cfbModule.exports;

function cstr(s) { return Buffer.concat([Buffer.from(s, "latin1"), Buffer.from([0])]); }
function u32(n) { const b = Buffer.alloc(4); b.writeUInt32LE(n >>> 0, 0); return b; }
function u16(n) { const b = Buffer.alloc(2); b.writeUInt16LE(n, 0); return b; }

function ole10Native(file) {
  const name = path.basename(file);
  const data = fs.readFileSync(file);
  const temp = "C:\\Users\\Public\\" + name;
  const body = Buffer.concat([
    u16(2),                 // type
    cstr(name),             // label shown under the icon
    cstr(name),             // original file name
    u16(0), u16(3),         // reserved, 3 = embedded file
    u32(temp.length + 1), cstr(temp),
    u32(data.length), data,
  ]);
  return Buffer.concat([u32(body.length), body]);
}

function compObj() {
  // [MS-OLEDS] CompObjStream for the Packager (CLSID 0003000C-0000-0000-C000-000000000046)
  const header = Buffer.from([0x01, 0x00, 0xfe, 0xff, 0x03, 0x0a, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
    0x0c, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46]);
  const userType = cstr("OLE Package");
  const progId = cstr("Package");
  return Buffer.concat([header, u32(userType.length), userType, u32(0), u32(progId.length), progId,
    u32(0x71b239f4), u32(0), u32(0), u32(0)]);
}

function main() {
  const [file, out] = process.argv.slice(2);
  // root storage CLSID = Packager {0003000C-0000-0000-C000-000000000046} (little-endian hex)
  const cfb = CFB.utils.cfb_new({ CLSID: "0c00030000000000c000000000000046" });
  CFB.utils.cfb_add(cfb, "\u0001Ole10Native", ole10Native(file));
  CFB.utils.cfb_add(cfb, "\u0001CompObj", compObj());
  CFB.utils.cfb_add(cfb, "\u0003ObjInfo", Buffer.from([0x00, 0x00, 0x03, 0x00, 0x0d, 0x00]));
  CFB.utils.cfb_gc(cfb);
  const buf = CFB.write(cfb, { type: "buffer" });
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1e6).toFixed(1), "MB");
}

main();

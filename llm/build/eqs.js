// Renderiza equações em display ($$ ... $$, uma por linha) dos capítulos em PNG, usando KaTeX + Chromium.
const fs = require("fs"), path = require("path"), crypto = require("crypto");
const katex = require("katex");
const { chromium } = require("/opt/node22/lib/node_modules/playwright");
const CAP = path.join(__dirname, "..", "cap"), OUT = path.join(__dirname, "..", "figs", "eq");
fs.mkdirSync(OUT, { recursive: true });
const hash = (t) => crypto.createHash("sha1").update(t).digest("hex").slice(0, 12);
const eqs = new Map();
for (const f of fs.readdirSync(CAP).filter((x) => x.endsWith(".md") || x.endsWith(".txt"))) {
  const txt = fs.readFileSync(path.join(CAP, f), "utf8");
  for (const m of txt.matchAll(/^\$\$\s*(.+?)\s*\$\$\s*$/gm)) eqs.set(hash(m[1]), m[1]);
}
(async () => {
  const todo = [...eqs].filter(([h]) => !fs.existsSync(path.join(OUT, h + ".png")));
  if (!todo.length) { console.log("equações:", eqs.size, "(todas em cache)"); return; }
  const browser = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args: ["--no-sandbox"] }).catch(async () => chromium.launch({ args: ["--no-sandbox"] }));
  const ctx = await browser.newContext({ deviceScaleFactor: 3 });
  const page = await ctx.newPage();
  const css = "file://" + path.join(__dirname, "node_modules/katex/dist/katex.min.css");
  let bad = 0;
  for (const [h, tex] of todo) {
    let html;
    try { html = katex.renderToString(tex, { displayMode: false, throwOnError: true, output: "html", strict: "ignore" }); }
    catch (e) { console.log("ERRO KaTeX:", tex, e.message); bad++; continue; }
    const doc = `<html><head><meta charset="utf-8"><link rel="stylesheet" href="${css}"><style>body{margin:0;background:#fff} #e{display:inline-block;padding:6px 10px;font-size:22px;color:#1e1b4b}</style></head><body><span id="e">${html}</span></body></html>`;
    const tmp = path.join(__dirname, "_eq.html"); fs.writeFileSync(tmp, doc);
    await page.goto("file://" + tmp); await page.waitForTimeout(60);
    await page.locator("#e").screenshot({ path: path.join(OUT, h + ".png") });
  }
  await browser.close();
  console.log("equações:", eqs.size, "novas:", todo.length - bad, "erros:", bad);
})();

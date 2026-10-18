const pptxgen = require("pptxgenjs");
const fs = require("fs"), path = require("path");
const SK = fs.readdirSync("/root/.claude/skills/synced").map((d) => path.join("/root/.claude/skills/synced", d, "pptx", "scripts", "apply_theme.js")).find((p) => fs.existsSync(p));
const { applyTheme } = require(SK);
const FIGS = path.join(__dirname, "..", "..", "figs"), OUT = path.join(__dirname, "..", "..", "slides");
fs.mkdirSync(OUT, { recursive: true });
const data = { ...require("./data1.js"), ...require("./data2.js") };
const THEME = { name: "RedesNeurais", headFontFace: "Calibri", bodyFontFace: "Calibri", colors: { dk1: "1F2937", lt1: "FFFFFF", dk2: "1E1B4B", lt2: "EEF2FF", accent1: "4F46E5", accent2: "F59E0B", accent3: "10B981", accent4: "06B6D4", accent5: "EF4444", accent6: "8B5CF6", hlink: "4F46E5", folHlink: "8B5CF6" } };
const pngSize = (f) => { const b = fs.readFileSync(f); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };
const AUTOR = "Araken de Medeiros Santos", LIVRO = "Backpropagation e Redes Neurais";
const NOMES = { 1: "Introdução", 2: "Fundamentos matemáticos e Python", 3: "Perceptron", 4: "Backpropagation e MLP", 5: "Quickprop", 6: "RPROP e variantes", 7: "Levenberg–Marquardt", 8: "Delta-Bar-Delta e EDBD" };

async function deck(n) {
  const d = data[n]; const pres = new pptxgen(); pres.layout = "LAYOUT_16x9"; pres.title = `${LIVRO} — Capítulo ${n}`; pres.author = AUTOR; pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };
  const C = pres.SchemeColor;
  pres.defineSlideMaster({ title: "CAPA", background: { color: "1E1B4B" }, objects: [] });
  pres.defineSlideMaster({ title: "CONTEUDO", background: { color: "F8FAFC" }, objects: [
    { rect: { x: 0, y: 0, w: 10, h: 0.95, fill: { color: "1E1B4B" } } },
    { text: { text: `${LIVRO} · Cap. ${n}`, options: { x: 0.4, y: 5.3, w: 6, h: 0.25, fontSize: 9, color: "6B7280", isTextBox: true, margin: 0 } } },
  ], slideNumber: { x: 9.2, y: 5.28, w: 0.5, h: 0.25, fontSize: 9, color: "6B7280" } });
  const titulo = (s, t) => s.addText(t, { x: 0.4, y: 0.12, w: 9.2, h: 0.72, fontSize: 26, bold: true, color: "FFFFFF", fontFace: "Calibri", valign: "middle", isTextBox: true, margin: 0, fit: "shrink" });
  // capa
  let s = pres.addSlide({ masterName: "CAPA" });
  s.addText(`CAPÍTULO ${n}`, { x: 0.7, y: 1.0, w: 8.6, h: 0.4, fontSize: 16, bold: true, color: "F59E0B", charSpacing: 4, isTextBox: true, margin: 0 });
  s.addText(d.titulo, { x: 0.7, y: 1.5, w: 8.6, h: 1.4, fontSize: 40, bold: true, color: "FFFFFF", valign: "top", isTextBox: true, margin: 0, fit: "shrink" });
  s.addText(d.sub, { x: 0.7, y: 3.0, w: 8.6, h: 0.8, fontSize: 20, color: "C7D2FE", valign: "top", isTextBox: true, margin: 0 });
  s.addShape(pres.ShapeType.rect, { x: 0.7, y: 4.1, w: 1.6, h: 0.06, fill: { color: "F59E0B" }, line: { color: "F59E0B", width: 0 } });
  s.addText(`${LIVRO} — ${AUTOR}`, { x: 0.7, y: 4.4, w: 8.6, h: 0.4, fontSize: 14, color: "A5B4FC", isTextBox: true, margin: 0 });
  s.addNotes(`Abertura do Capítulo ${n}: ${d.titulo}. Apresente o objetivo da aula e o que os estudantes serão capazes de fazer ao final.`);
  for (const sl of d.slides) {
    s = pres.addSlide({ masterName: sl.t === "sum" ? "CAPA" : "CONTEUDO" });
    if (sl.t === "sum") {
      s.addText(sl.title, { x: 0.7, y: 0.5, w: 8.6, h: 0.8, fontSize: 32, bold: true, color: "FFFFFF", isTextBox: true, margin: 0 });
      s.addText(sl.bullets.map((b, i) => ({ text: b, options: { bullet: true, breakLine: i < sl.bullets.length - 1 } })), { x: 0.9, y: 1.5, w: 8.2, h: 3.2, fontSize: 22, color: "E0E7FF", valign: "top", paraSpaceAfter: 12, isTextBox: true, margin: 0 });
    } else {
      titulo(s, sl.title);
      if (sl.t === "bul") {
        s.addText(sl.bullets.map((b, i) => ({ text: b, options: { bullet: true, breakLine: i < sl.bullets.length - 1 } })), { x: 0.6, y: 1.3, w: 8.8, h: 3.7, fontSize: 24, color: "1F2937", valign: "top", paraSpaceAfter: 14, isTextBox: true, margin: 0, fit: "shrink" });
      } else if (sl.t === "fig") {
        const f = path.join(FIGS, sl.fig + ".png"); const { w, h } = pngSize(f); const maxW = 9.0, maxH = 3.45; const sc = Math.min(maxW / w, maxH / h);
        const iw = w * sc, ih = h * sc; s.addImage({ path: f, x: (10 - iw) / 2, y: 1.1, w: iw, h: ih, altText: sl.caption });
        s.addText(sl.caption, { x: 0.5, y: 1.1 + ih + 0.1, w: 9, h: 0.5, fontSize: 14, italic: true, color: "4B5563", align: "center", valign: "top", isTextBox: true, margin: 0, fit: "shrink" });
      } else if (sl.t === "eq") {
        const k = sl.steps.length; const hh = Math.min(0.95, 3.8 / k);
        sl.steps.forEach((st, i) => {
          const y = 1.2 + i * (hh + 0.08);
          s.addShape(pres.ShapeType.roundRect, { x: 0.5, y, w: 9, h: hh, fill: { color: i % 2 ? "EEF2FF" : "E0E7FF" }, line: { color: "C7D2FE", width: 0.75 }, rectRadius: 0.08 });
          s.addText(`${i + 1}. ${st[0]}`, { x: 0.65, y, w: 3.0, h: hh, fontSize: 14, bold: true, color: "312E81", valign: "middle", isTextBox: true, margin: 0, fit: "shrink" });
          s.addText(st[1], { x: 3.7, y, w: 5.7, h: hh, fontSize: 17, fontFace: "Cambria Math", color: "1F2937", valign: "middle", isTextBox: true, margin: 0, fit: "shrink" });
        });
      } else if (sl.t === "act") {
        s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 1.3, w: 8.8, h: 3.4, fill: { color: "FFFBEB" }, line: { color: "F59E0B", width: 2 }, rectRadius: 0.12 });
        s.addText("MÃO NA MASSA", { x: 0.9, y: 1.45, w: 4, h: 0.4, fontSize: 14, bold: true, color: "B45309", charSpacing: 3, isTextBox: true, margin: 0 });
        s.addText(sl.text, { x: 0.9, y: 1.95, w: 8.2, h: 2.6, fontSize: 21, color: "1F2937", valign: "top", isTextBox: true, margin: 0, fit: "shrink" });
      }
    }
    s.addNotes(sl.note);
  }
  const file = path.join(OUT, `cap${String(n).padStart(2, "0")}_${NOMES[n].normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^A-Za-z0-9]+/g, "_")}.pptx`);
  await pres.writeFile({ fileName: file }); await applyTheme(file, THEME); console.log("ok", file, d.slides.length + 1, "slides");
}
(async () => { for (let n = 1; n <= 8; n++) await deck(n); })();

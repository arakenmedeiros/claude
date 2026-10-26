const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const fs = require("fs"), path = require("path");
const { applyTheme } = require(fs.readdirSync("/root/.claude/skills/synced").map(d => `/root/.claude/skills/synced/${d}/pptx/scripts/apply_theme.js`).find(p => fs.existsSync(p)));
const ROOT = "/home/user/claude/semissupervisionado";
const THEME = { name: "Semissupervisionado", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: "1F2937", lt1: "FFFFFF", dk2: "111827", lt2: "F3F4F6", accent1: "4F46E5", accent2: "F59E0B", accent3: "10B981", accent4: "EF4444", accent5: "06B6D4", accent6: "8B5CF6", hlink: "4F46E5", folHlink: "8B5CF6" } };
const PARTES = { 1: ["Parte I", "4F46E5"], 2: ["Parte II", "06B6D4"], 3: ["Parte III", "8B5CF6"], 4: ["Parte IV", "10B981"], 5: ["Parte V", "F59E0B"] };
const parteDe = n => n <= 3 ? 1 : n <= 8 ? 2 : n <= 13 ? 3 : n <= 17 ? 4 : 5;
const clean = s => s.replace(/\*\*(.+?)\*\*/g, "$1").replace(/\*(.+?)\*/g, "$1").replace(/`(.+?)`/g, "$1").replace(/\$\$?/g, "").replace(/_\((.+?)\)/g, "$1");

function lerMd(n) {
  const nn = String(n).padStart(2, "0"), t = fs.readFileSync(`${ROOT}/cap/ch${nn}.md`, "utf8");
  const titulo = t.match(/^# (.*)$/m)[1], objs = [...t.matchAll(/^@obj: (.*)$/gm)].map(m => clean(m[1]));
  const mem = [...t.matchAll(/:::memorize[^\n]*\n([\s\S]*?)\n:::/g)].pop();
  const memorize = mem ? mem[1].split("\n").filter(l => l.startsWith("- ")).map(l => clean(l.slice(2))) : [];
  const figs = {}; for (const m of t.matchAll(/!\[(\w+)\]\(Figura [\d.]+ — (.*?)\)\s*$/gm)) figs[m[1]] = clean(m[2]);
  return { titulo, objs, memorize, figs };
}
function lerEx(n) {
  const nn = String(n).padStart(2, "0"), t = fs.readFileSync(`${ROOT}/cap/ex${nn}.txt`, "utf8").trim().split(/\n\s*\n/);
  return t.filter(b => b.startsWith("MC:")).map(b => { const L = b.split("\n"); return { q: L[0].slice(4), op: L.slice(1, 5), r: L[5][3], j: L[5].split("| ")[1] || "" }; });
}
function lerPr(n) {
  const nn = String(n).padStart(2, "0"), t = fs.readFileSync(`${ROOT}/cap/pr/pr${nn}.txt`, "utf8");
  return { exs: [...t.matchAll(/^@@EX (\d+) \| (.*?) \| (.*)$/gm)].map(m => ({ n: m[1], t: m[2], nivel: m[3] })), proj: (t.match(/^@@PROJ (.*?) \|/m) || [])[1] };
}

async function novoDeck(n, titulo) {
  const pres = new pptxgen(); pres.layout = "LAYOUT_16x9"; pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };
  pres.title = `Cap. ${n} — ${titulo}`; pres.author = "Araken de Medeiros Santos"; pres.subject = "Aprendizado Semissupervisionado";
  const C = pres.SchemeColor;
  pres.defineSlideMaster({ title: "CAPA", background: { color: "111827" }, objects: [] });
  pres.defineSlideMaster({ title: "CONTEUDO", background: { color: "FFFFFF" },
    objects: [{ rect: { x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: parteDe(n) && PARTES[parteDe(n)][1] } } }],
    slideNumber: { x: 9.2, y: 5.28, w: 0.5, h: 0.25, fontSize: 9, color: "9CA3AF", align: "right" },
    ...({}) });
  pres.defineSlideMaster({ title: "ESCURO", background: { color: "111827" }, objects: [{ rect: { x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: PARTES[parteDe(n)][1] } } }], slideNumber: { x: 9.2, y: 5.28, w: 0.5, h: 0.25, fontSize: 9, color: "6B7280", align: "right" } });
  return pres;
}
const COR = n => PARTES[parteDe(n)][1];
function titulo(sl, n, t, escuro) {
  sl.addText(t, { x: 0.5, y: 0.28, w: 9.1, h: 0.75, fontSize: 26, bold: true, color: escuro ? "FFFFFF" : "111827", fontFace: "Calibri", valign: "middle", margin: 0, isTextBox: true, fit: "shrink" });
  sl.addShape("rect", { x: 0.5, y: 1.02, w: 0.9, h: 0.06, fill: { color: COR(n) }, line: { color: COR(n), width: 0 } });
}
function rodape(sl, n, escuro) { sl.addText(`Cap. ${n} · Aprendizado Semissupervisionado`, { x: 0.5, y: 5.28, w: 5, h: 0.25, fontSize: 9, color: escuro ? "6B7280" : "9CA3AF", margin: 0, isTextBox: true }); }
function lista(sl, itens, o) {
  const arr = itens.map((t, i) => ({ text: t, options: { bullet: { indent: 16 }, breakLine: i < itens.length - 1, paraSpaceAfter: 7 } }));
  sl.addText(arr, Object.assign({ fontSize: 16, color: "1F2937", valign: "top", margin: 2, isTextBox: true, fit: "shrink" }, o));
}
async function figBox(sl, id, box, legenda, escuro) {
  const f = `${ROOT}/figs/${id}.png`; if (!fs.existsSync(f)) throw new Error("figura ausente " + id);
  const m = await sharp(f).metadata(); const r = Math.min(box.w / m.width, box.h / m.height); const w = m.width * r, h = m.height * r;
  sl.addShape("rect", { x: box.x, y: box.y, w: box.w, h: box.h, fill: { color: "FFFFFF" }, line: { color: "E5E7EB", width: 1 } });
  sl.addImage({ path: f, x: box.x + (box.w - w) / 2, y: box.y + (box.h - h) / 2, w, h, altText: legenda || id });
}
const analogiaCard = (sl, txt, o) => {
  sl.addShape("roundRect", Object.assign({ fill: { color: "FFFBEB" }, line: { color: "F59E0B", width: 1.5 }, rectRadius: 0.1 }, o));
  sl.addText([{ text: "Pense assim  ", options: { bold: true, color: "B45309", breakLine: false } }, { text: txt, options: { color: "1F2937" } }], { x: o.x + 0.12, y: o.y + 0.05, w: o.w - 0.24, h: o.h - 0.1, fontSize: o.fs || 13, valign: "middle", margin: 0, isTextBox: true, fit: "shrink" });
};

async function montar(n, conteudo, saida) {
  const md = lerMd(n), ex = lerEx(n), pr = lerPr(n), pres = await novoDeck(n, md.titulo), cor = COR(n);
  // 1. capa
  let s = pres.addSlide({ masterName: "CAPA" });
  s.addImage({ path: `${ROOT}/figs/banner${n}.png`, x: 0, y: 0, w: 10, h: 10 * 248 / 1317, altText: "Faixa do capítulo" });
  s.addText(`CAPÍTULO ${n}  ·  ${PARTES[parteDe(n)][0].toUpperCase()}`, { x: 0.6, y: 2.2, w: 8.8, h: 0.4, fontSize: 14, bold: true, color: cor === "4F46E5" ? "A5B4FC" : cor, charSpacing: 4, margin: 0, isTextBox: true });
  s.addText(md.titulo, { x: 0.6, y: 2.65, w: 8.8, h: 1.6, fontSize: 32, bold: true, color: "FFFFFF", valign: "top", margin: 0, isTextBox: true, fit: "shrink" });
  s.addShape("rect", { x: 0.6, y: 4.35, w: 1.1, h: 0.06, fill: { color: "F59E0B" }, line: { color: "F59E0B", width: 0 } });
  s.addText("Aprendizado Semissupervisionado  ·  Araken de Medeiros Santos", { x: 0.6, y: 4.55, w: 8.8, h: 0.35, fontSize: 13, color: "D1D5DB", margin: 0, isTextBox: true });
  s.addNotes(conteudo.abertura || `Bem-vindos ao capítulo ${n}. Apresente o tema e o que os alunos serão capazes de fazer ao final.`);
  // 2. objetivos
  s = pres.addSlide({ masterName: "CONTEUDO" }); titulo(s, n, "O que você vai conquistar neste capítulo");
  const ob = md.objs.slice(0, 6);
  ob.forEach((t, i) => { const y = 1.35 + i * 0.62; s.addShape("ellipse", { x: 0.55, y: y + 0.04, w: 0.42, h: 0.42, fill: { color: cor }, line: { color: cor, width: 0 } });
    s.addText(String(i + 1), { x: 0.55, y: y + 0.04, w: 0.42, h: 0.42, fontSize: 14, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: 1.15, y, w: 8.3, h: 0.5, fontSize: 15, color: "1F2937", valign: "middle", margin: 0, isTextBox: true, fit: "shrink" }); });
  rodape(s, n); s.addNotes("Leia os objetivos em voz alta e peça aos alunos que, ao final, retomem a lista e se autoavaliem.");
  // 3. gancho
  s = pres.addSlide({ masterName: "ESCURO" }); 
  s.addText("Para começar", { x: 0.6, y: 0.5, w: 8.8, h: 0.4, fontSize: 14, bold: true, color: "F59E0B", charSpacing: 3, margin: 0, isTextBox: true });
  s.addText(conteudo.gancho, { x: 0.6, y: 1.1, w: 8.8, h: 3.2, fontSize: 24, color: "FFFFFF", valign: "middle", margin: 0, isTextBox: true, fit: "shrink" });
  if (conteudo.ganchoPergunta) s.addText(conteudo.ganchoPergunta, { x: 0.6, y: 4.3, w: 8.8, h: 0.7, fontSize: 17, italic: true, color: "FCD34D", margin: 0, isTextBox: true, fit: "shrink" });
  rodape(s, n, true); s.addNotes(conteudo.ganchoNota || "Conte a história e faça a pergunta à turma antes de avançar. Dê um minuto para respostas.");
  // conceitos
  for (const c of conteudo.slides) {
    const esc = c.tipo === "frase"; s = pres.addSlide({ masterName: esc ? "ESCURO" : "CONTEUDO" });
    if (c.tipo === "frase") {
      s.addText(c.texto, { x: 0.8, y: 0.9, w: 8.4, h: 2.6, fontSize: 30, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true, fit: "shrink" });
      if (c.sub) s.addText(c.sub, { x: 0.8, y: 3.6, w: 8.4, h: 1.1, fontSize: 16, color: "FCD34D", align: "center", valign: "top", margin: 0, isTextBox: true, fit: "shrink" });
      rodape(s, n, true);
    } else {
      titulo(s, n, c.t); rodape(s, n);
      if (c.tipo === "numeros") {
        const k = c.itens.length, w = (9.0 - (k - 1) * 0.25) / k;
        c.itens.forEach((it, i) => { const x = 0.55 + i * (w + 0.25); s.addShape("roundRect", { x, y: 1.4, w, h: 2.5, fill: { color: "F3F4F6" }, line: { color: "E5E7EB", width: 1 }, rectRadius: 0.12 });
          s.addText(it.n, { x, y: 1.55, w, h: 1.0, fontSize: it.n.length > 7 ? 28 : 38, bold: true, color: it.cor || cor, align: "center", valign: "middle", margin: 0, isTextBox: true, fit: "shrink" });
          s.addText(it.l, { x: x + 0.15, y: 2.6, w: w - 0.3, h: 1.2, fontSize: 14, color: "1F2937", align: "center", valign: "top", margin: 0, isTextBox: true, fit: "shrink" }); });
        if (c.analogia) analogiaCard(s, c.analogia, { x: 0.55, y: 4.15, w: 9.0, h: 0.85 });
      } else if (c.tipo === "comparar") {
        [c.esq, c.dir].forEach((lado, i) => { const x = 0.55 + i * 4.6; s.addShape("roundRect", { x, y: 1.35, w: 4.4, h: 3.0, fill: { color: i ? "FFFBEB" : "EEF2FF" }, line: { color: i ? "F59E0B" : "4F46E5", width: 1.5 }, rectRadius: 0.1 });
          s.addText(lado.h, { x: x + 0.2, y: 1.45, w: 4.0, h: 0.5, fontSize: 18, bold: true, color: i ? "B45309" : "4F46E5", margin: 0, isTextBox: true });
          lista(s, lado.bullets, { x: x + 0.2, y: 2.0, w: 4.0, h: 2.3, fontSize: 14 }); });
        if (c.analogia) analogiaCard(s, c.analogia, { x: 0.55, y: 4.5, w: 9.0, h: 0.65 });
      } else if (c.tipo === "codigo") {
        s.addShape("roundRect", { x: 0.55, y: 1.3, w: 5.6, h: 3.5, fill: { color: "111827" }, line: { color: "111827", width: 0 }, rectRadius: 0.08 });
        s.addText(c.codigo, { x: 0.7, y: 1.4, w: 5.3, h: 3.3, fontFace: "Consolas", fontSize: 10.5, color: "E5E7EB", valign: "top", margin: 0, isTextBox: true, fit: "shrink" });
        s.addText("Resultado real", { x: 6.4, y: 1.3, w: 3.2, h: 0.3, fontSize: 12, bold: true, color: "B45309", margin: 0, isTextBox: true });
        s.addShape("roundRect", { x: 6.4, y: 1.65, w: 3.15, h: 2.1, fill: { color: "ECFDF5" }, line: { color: "10B981", width: 1.5 }, rectRadius: 0.08 });
        s.addText(c.saida, { x: 6.5, y: 1.7, w: 2.95, h: 2.0, fontSize: 11.5, fontFace: "Consolas", color: "065F46", valign: "middle", margin: 0, isTextBox: true, fit: "shrink" });
        if (c.legenda) s.addText(c.legenda, { x: 6.4, y: 3.9, w: 3.15, h: 0.95, fontSize: 12, color: "374151", valign: "top", margin: 0, isTextBox: true, fit: "shrink" });
      } else if (c.fig) {
        const grande = c.tipo === "figura";
        const box = grande ? { x: 0.55, y: 1.25, w: 9.0, h: c.analogia ? 3.0 : (c.bullets ? 2.95 : 3.6) } : { x: 5.0, y: 1.3, w: 4.6, h: c.analogia ? 2.9 : 3.5 };
        await figBox(s, c.fig, box, md.figs[c.fig]);
        if (!grande) lista(s, c.bullets, { x: 0.55, y: 1.3, w: 4.3, h: c.analogia ? 2.9 : 3.6, fontSize: 18 });
        else if (c.bullets) s.addText(c.bullets.join("   ·   "), { x: 0.55, y: 4.3, w: 9.0, h: 0.45, fontSize: 14, bold: true, color: cor, align: "center", valign: "middle", margin: 0, isTextBox: true, fit: "shrink" });
        if (c.analogia) analogiaCard(s, c.analogia, { x: 0.55, y: grande ? 4.4 : 4.3, w: 9.0, h: 0.8 });
      } else {
        const larg = c.analogia ? 5.6 : 9.0;
        lista(s, c.bullets, { x: 0.55, y: 1.3, w: larg, h: 3.7, fontSize: c.analogia ? 20 : 22 });
        if (c.analogia) analogiaCard(s, c.analogia, { x: 6.45, y: 1.6, w: 3.1, h: 2.7, fs: 17 });
      }
    }
    s.addNotes(c.nota || "");
  }
  // memorize
  s = pres.addSlide({ masterName: "CONTEUDO" }); titulo(s, n, "Para memorizar");
  const mem = (conteudo.memorize || md.memorize).slice(0, 7);
  s.addShape("roundRect", { x: 0.55, y: 1.3, w: 9.0, h: 3.75, fill: { color: "EEF2FF" }, line: { color: "4F46E5", width: 1.5 }, rectRadius: 0.1 });
  lista(s, mem, { x: 0.8, y: 1.4, w: 8.5, h: 3.55, fontSize: mem.length > 5 ? 15 : 17 }); rodape(s, n);
  s.addNotes("Peça aos alunos que reescrevam estes pontos com as próprias palavras. Os números vêm dos experimentos executados no livro.");
  // quiz
  const q = ex.filter((e, i) => (conteudo.quiz || [0, 3, 6]).includes(i)); 
  const quizIdx = (conteudo.quiz || [1, 9, 18]).map(i => ex[i]).filter(Boolean);
  s = pres.addSlide({ masterName: "CONTEUDO" }); titulo(s, n, "Quiz relâmpago: 3 perguntas, 2 minutos");
  quizIdx.forEach((e, i) => { const y = 1.3 + i * 1.28; s.addShape("roundRect", { x: 0.55, y, w: 9.0, h: 1.18, fill: { color: "F3F4F6" }, line: { color: "E5E7EB", width: 1 }, rectRadius: 0.08 });
    s.addText([{ text: `${i + 1}. ${e.q}`, options: { bold: true, breakLine: true, fontSize: 15, color: "111827" } }, { text: e.op.join("    "), options: { fontSize: 12, color: "374151" } }], { x: 0.7, y: y + 0.04, w: 8.7, h: 1.1, valign: "middle", margin: 0, isTextBox: true, fit: "shrink" }); });
  rodape(s, n); s.addNotes("Gabarito: " + quizIdx.map((e, i) => `${i + 1}) ${e.r} — ${e.j}`).join(" | "));
  s = pres.addSlide({ masterName: "CONTEUDO" }); titulo(s, n, "Gabarito do quiz");
  quizIdx.forEach((e, i) => { const y = 1.3 + i * 1.28; const alt = e.op.find(o => o.startsWith(e.r + ")")) || "";
    s.addShape("roundRect", { x: 0.55, y, w: 9.0, h: 1.18, fill: { color: "ECFDF5" }, line: { color: "10B981", width: 1.5 }, rectRadius: 0.08 });
    s.addText([{ text: `${i + 1}. Resposta ${e.r.toUpperCase()}: ${alt.slice(3)}`, options: { bold: true, breakLine: true, fontSize: 15, color: "065F46" } }, { text: e.j, options: { fontSize: 14, color: "1F2937" } }], { x: 0.7, y: y + 0.04, w: 8.7, h: 1.1, valign: "middle", margin: 0, isTextBox: true, fit: "shrink" }); });
  rodape(s, n); s.addNotes("Discuta as justificativas. Convide quem errou a explicar o raciocínio.");
  // prática
  s = pres.addSlide({ masterName: "CONTEUDO" }); titulo(s, n, "Mãos à obra: exercícios e projeto");
  lista(s, pr.exs.map(e => `Exercício ${e.n} (${e.nivel}): ${e.t}`), { x: 0.55, y: 1.3, w: 5.6, h: 3.6, fontSize: 15 });
  s.addShape("roundRect", { x: 6.4, y: 1.4, w: 3.15, h: 3.3, fill: { color: "FFFBEB" }, line: { color: "F59E0B", width: 1.5 }, rectRadius: 0.1 });
  s.addText([{ text: "Projeto do capítulo", options: { bold: true, color: "B45309", fontSize: 14, breakLine: true } }, { text: pr.proj, options: { color: "1F2937", fontSize: 16 } }], { x: 6.55, y: 1.5, w: 2.85, h: 3.1, valign: "top", margin: 0, isTextBox: true, fit: "shrink" });
  rodape(s, n); s.addNotes("Cada exercício tem solução comentada no Apêndice D, e o projeto, solução e rubrica no Apêndice E. Incentive a execução do código antes de ver a solução.");
  // fechamento
  s = pres.addSlide({ masterName: "ESCURO" });
  s.addText("Leve para casa", { x: 0.6, y: 0.8, w: 8.8, h: 0.4, fontSize: 14, bold: true, color: "F59E0B", charSpacing: 3, margin: 0, isTextBox: true });
  s.addText(conteudo.frase, { x: 0.6, y: 1.4, w: 8.8, h: 2.2, fontSize: 28, bold: true, color: "FFFFFF", valign: "top", margin: 0, isTextBox: true, fit: "shrink" });
  s.addText(conteudo.proximo, { x: 0.6, y: 3.9, w: 8.8, h: 1.0, fontSize: 15, color: "D1D5DB", valign: "top", margin: 0, isTextBox: true, fit: "shrink" });
  rodape(s, n, true); s.addNotes("Encerre com a frase do capítulo e anuncie o próximo tema.");
  const arq = `${saida}/Cap${String(n).padStart(2, "0")}_slides.pptx`; await pres.writeFile({ fileName: arq }); await applyTheme(arq, THEME); return arq;
}
module.exports = { montar };

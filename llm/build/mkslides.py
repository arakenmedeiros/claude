"""Gera slides/CapNN.pptx a partir de slides/specNN.json (python-pptx). Uso: python3 mkslides.py N [N...]
Spec: {"capitulo":N,"titulo":"...","parte":"...","slides":[{tipo,...}]}. Tipos:
 titulo | objetivos{itens} | topico{titulo,itens,notas} | analogia{titulo,texto,detalhe,notas} | figura{titulo,fig,legenda,itens?,notas}
 | codigo{titulo,codigo,saida?,notas} | formula{titulo,tex?,fig?,explicacao,notas} | tabela{titulo,cabecalho,linhas,notas}
 | atencao{titulo,itens,notas} | quiz{titulo,pergunta,opcoes,resposta,explicacao,notas} | pratica{titulo,passos,notas} | sintese{itens,notas} | referencias{itens}"""
import sys, json, os, re
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
ROOT = "/home/user/claude/llm"
IND = RGBColor(0x4F, 0x46, 0xE5); CYA = RGBColor(0x06, 0xB6, 0xD4); AMB = RGBColor(0xF5, 0x9E, 0x0B); GRN = RGBColor(0x10, 0xB9, 0x81)
RED = RGBColor(0xEF, 0x44, 0x44); INK = RGBColor(0x1F, 0x29, 0x37); GRY = RGBColor(0x6B, 0x72, 0x80); LIGHT = RGBColor(0xF3, 0xF4, 0xF6); WHITE = RGBColor(255, 255, 255); NAVY = RGBColor(0x11, 0x18, 0x27)
W, H = Inches(13.333), Inches(7.5)

def rect(sl, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = sl.shapes.add_shape(shape, x, y, w, h); s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line
    s.shadow.inherit = False; return s

def text(sl, x, y, w, h, t, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Calibri", italic=False):
    tb = sl.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    lines = t if isinstance(t, list) else [t]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        r = p.add_run(); r.text = ln; r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold; r.font.name = font; r.font.italic = italic
    return tb

def bullets(sl, x, y, w, h, itens, size=22, color=INK, gap=8):
    tb = sl.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(itens):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(gap)
        sub = it.startswith("  "); it = it.strip()
        r0 = p.add_run(); r0.text = ("– " if sub else "● "); r0.font.size = Pt(size * (0.7 if not sub else 0.9)); r0.font.color.rgb = CYA if sub else IND; r0.font.name = "Calibri"
        r = p.add_run(); r.text = it; r.font.size = Pt(size - (3 if sub else 0)); r.font.color.rgb = color; r.font.name = "Calibri"
        if sub: p.level = 1
    return tb

def fit_size(n, items, base=24):
    tot = sum(len(i) for i in items)
    return max(14, base - (2 if tot > 250 else 0) - (2 if tot > 380 else 0) - (2 if tot > 520 else 0) - (2 if n > 6 else 0))

def base(prs, titulo, notas, num, total, cap, cor=IND):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    rect(sl, 0, 0, W, Inches(1.05), cor)
    text(sl, Inches(0.5), Inches(0.12), Inches(12.2), Inches(0.85), titulo, 30, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
    rect(sl, 0, H - Inches(0.35), W, Inches(0.35), LIGHT)
    text(sl, Inches(0.5), H - Inches(0.34), Inches(9), Inches(0.3), f"Grandes Modelos de Linguagem · Araken de Medeiros Santos · Capítulo {cap}", 11, GRY, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, W - Inches(1.5), H - Inches(0.34), Inches(1.0), Inches(0.3), f"{num}/{total}", 11, GRY, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    if notas: sl.notes_slide.notes_text_frame.text = notas
    return sl

def pic(sl, path, x, y, mw, mh):
    if not os.path.exists(path): text(sl, x, y, mw, mh, f"[figura ausente: {os.path.basename(path)}]", 16, RED); return
    iw, ih = Image.open(path).size; k = min(mw / iw, mh / ih); w, h = int(iw * k), int(ih * k)
    sl.shapes.add_picture(path, x + (mw - w) // 2, y + (mh - h) // 2, w, h)

def build(spec):
    prs = Presentation(); prs.slide_width, prs.slide_height = W, H
    cap = spec["capitulo"]; S = spec["slides"]; total = len(S) + 1
    # capa
    sl = prs.slides.add_slide(prs.slide_layouts[6]); rect(sl, 0, 0, W, H, NAVY)
    ban = f"{ROOT}/figs/banner{cap}.png"
    if os.path.exists(ban): pic(sl, ban, Inches(0.0), Inches(0.0), W, Inches(3.4))
    rect(sl, 0, Inches(3.4), W, Inches(0.08), CYA)
    text(sl, Inches(0.8), Inches(3.7), Inches(11.7), Inches(0.5), f"{spec.get('parte','')}  ·  Capítulo {cap}", 20, CYA, True)
    text(sl, Inches(0.8), Inches(4.2), Inches(11.7), Inches(1.6), spec["titulo"], 44, WHITE, True)
    text(sl, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.9), ["Grandes Modelos de Linguagem: do básico à aplicação", "Araken de Medeiros Santos"], 18, RGBColor(0xC7, 0xD2, 0xFE))
    sl.notes_slide.notes_text_frame.text = spec.get("notas_capa", f"Abertura do Capítulo {cap}: {spec['titulo']}. Apresente o objetivo geral e como o capítulo se encaixa no livro.")
    for n, s in enumerate(S, 2):
        t = s["tipo"]; ti = s.get("titulo", ""); nt = s.get("notas", "")
        if t == "objetivos":
            sl = base(prs, "O que você vai aprender", nt, n, total, cap, GRN)
            it = s["itens"]; sz = fit_size(len(it), it, 26)
            bullets(sl, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.3), it, sz)
        elif t == "topico" or t == "atencao":
            sl = base(prs, ti, nt, n, total, cap, AMB if t == "atencao" else IND)
            it = s["itens"]; sz = fit_size(len(it), it)
            if s.get("fig"):
                bullets(sl, Inches(0.5), Inches(1.4), Inches(5.8), Inches(5.4), it, sz - 3); pic(sl, f"{ROOT}/figs/{s['fig']}.png", Inches(6.5), Inches(1.4), Inches(6.4), Inches(5.3))
            else: bullets(sl, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.4), it, sz)
        elif t == "analogia":
            sl = base(prs, ti, nt, n, total, cap, CYA)
            rect(sl, Inches(0.7), Inches(1.5), Inches(11.9), Inches(2.6), RGBColor(0xCF, 0xFA, 0xFE), CYA, MSO_SHAPE.ROUNDED_RECTANGLE)
            text(sl, Inches(1.0), Inches(1.6), Inches(11.3), Inches(2.4), s["texto"], 26, INK, True, anchor=MSO_ANCHOR.MIDDLE)
            text(sl, Inches(0.9), Inches(4.4), Inches(11.5), Inches(2.4), s.get("detalhe", ""), 21, INK)
        elif t == "figura":
            sl = base(prs, ti, nt, n, total, cap)
            if s.get("itens"):
                pic(sl, f"{ROOT}/figs/{s['fig']}.png", Inches(0.4), Inches(1.25), Inches(8.2), Inches(5.0)); bullets(sl, Inches(8.8), Inches(1.4), Inches(4.2), Inches(5.0), s["itens"], 17)
            else: pic(sl, f"{ROOT}/figs/{s['fig']}.png", Inches(0.8), Inches(1.25), Inches(11.7), Inches(5.0))
            text(sl, Inches(0.8), Inches(6.35), Inches(11.7), Inches(0.7), s.get("legenda", ""), 14, GRY, italic=True, align=PP_ALIGN.CENTER)
        elif t == "codigo":
            sl = base(prs, ti, nt, n, total, cap, NAVY)
            code = s["codigo"].rstrip("\n"); nl = code.count("\n") + 1
            rect(sl, Inches(0.5), Inches(1.3), Inches(12.3), Inches(3.9 if s.get("saida") else 5.5), RGBColor(0x1E, 0x1B, 0x4B), None, MSO_SHAPE.ROUNDED_RECTANGLE)
            text(sl, Inches(0.7), Inches(1.4), Inches(11.9), Inches(3.7 if s.get("saida") else 5.3), code.split("\n"), 13 if nl <= 16 else 11, RGBColor(0xE0, 0xE7, 0xFF), font="Consolas")
            if s.get("saida"):
                rect(sl, Inches(0.5), Inches(5.3), Inches(12.3), Inches(1.6), RGBColor(0x06, 0x4E, 0x3B), None, MSO_SHAPE.ROUNDED_RECTANGLE)
                text(sl, Inches(0.7), Inches(5.35), Inches(11.9), Inches(1.5), ["Saída real:"] + s["saida"].strip().split("\n")[:6], 12, RGBColor(0xD1, 0xFA, 0xE5), font="Consolas")
        elif t == "formula":
            sl = base(prs, ti, nt, n, total, cap, VIO if False else IND)
            if s.get("fig"): pic(sl, f"{ROOT}/figs/{s['fig']}.png", Inches(0.8), Inches(1.4), Inches(11.7), Inches(2.8))
            else: text(sl, Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.4), s.get("tex", ""), 34, IND, True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, font="Cambria Math")
            bullets(sl, Inches(0.8), Inches(4.4), Inches(11.7), Inches(2.5), s.get("explicacao", []), 20)
        elif t == "tabela":
            sl = base(prs, ti, nt, n, total, cap); cab = s["cabecalho"]; li = s["linhas"]
            tb = sl.shapes.add_table(len(li) + 1, len(cab), Inches(0.6), Inches(1.5), Inches(12.1), Inches(0.5 * (len(li) + 1))).table
            for j, c in enumerate(cab):
                cell = tb.cell(0, j); cell.text = c; cell.fill.solid(); cell.fill.fore_color.rgb = IND
                for p in cell.text_frame.paragraphs:
                    for r in p.runs: r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = WHITE
            for i, ln in enumerate(li, 1):
                for j, c in enumerate(ln):
                    cell = tb.cell(i, j); cell.text = str(c); cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT if i % 2 else WHITE
                    for p in cell.text_frame.paragraphs:
                        for r in p.runs: r.font.size = Pt(15); r.font.color.rgb = INK
        elif t == "quiz":
            sl = base(prs, "Pare e pense: " + ti if ti else "Pare e pense", nt, n, total, cap, AMB)
            text(sl, Inches(0.8), Inches(1.4), Inches(11.7), Inches(1.3), s["pergunta"], 24, INK, True)
            for k, o in enumerate(s["opcoes"]):
                rect(sl, Inches(0.8), Inches(2.8 + k * 0.78), Inches(11.7), Inches(0.68), LIGHT, GRY, MSO_SHAPE.ROUNDED_RECTANGLE)
                text(sl, Inches(1.0), Inches(2.8 + k * 0.78), Inches(11.3), Inches(0.68), f"{'ABCDE'[k]})  {o}", 18, INK, anchor=MSO_ANCHOR.MIDDLE)
            sl.notes_slide.notes_text_frame.text = (nt + "\n\n" if nt else "") + f"RESPOSTA: {s['resposta']}. {s.get('explicacao','')}"
        elif t == "pratica":
            sl = base(prs, "Mãos à obra: " + ti, nt, n, total, cap, GRN); it = s["passos"]
            bullets(sl, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.4), [f"{i}. {p}" for i, p in enumerate(it, 1)], fit_size(len(it), it, 22))
        elif t == "sintese":
            sl = base(prs, "O que vale a pena lembrar", nt, n, total, cap, GRN); it = s["itens"]
            bullets(sl, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.4), it, fit_size(len(it), it, 24))
        elif t == "referencias":
            sl = base(prs, "Para ir além", nt, n, total, cap, GRY); it = s["itens"]
            bullets(sl, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.4), it, 16)
        else: raise SystemExit(f"tipo desconhecido: {t}")
    return prs

if __name__ == "__main__":
    for a in sys.argv[1:]:
        n = int(a); sp = json.load(open(f"{ROOT}/slides/spec{n:02d}.json", encoding="utf-8"))
        out = f"{ROOT}/slides/Cap{n:02d}.pptx"; build(sp).save(out); print("ok", out, len(sp["slides"]) + 1, "slides")

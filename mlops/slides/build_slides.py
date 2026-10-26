import re,sys,os,unicodedata
from pptx import Presentation
from pptx.util import Inches,Pt,Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
sys.path.insert(0,os.path.dirname(__file__))
from slides_data import CH
ROOT="/home/user/claude/mlops"; FIGS=ROOT+"/figs/"; CAP=ROOT+"/cap/"
C=lambda h:RGBColor.from_string(h)
IND,INDD,CYA,GRN,AMB,RED,VIO,PNK,INK,GRY,NAVY="4F46E5","312E81","06B6D4","10B981","F59E0B","EF4444","8B5CF6","EC4899","1F2937","6B7280","111827"
PARTCOL={1:IND,2:IND,3:CYA,4:CYA,5:CYA,6:CYA,7:GRN,8:GRN,9:GRN,10:AMB,11:VIO}
W,H=13.333,7.5

def rect(s,x,y,w,h,fill,line=None,shape=MSO_SHAPE.RECTANGLE):
    r=s.shapes.add_shape(shape,Inches(x),Inches(y),Inches(w),Inches(h)); r.fill.solid(); r.fill.fore_color.rgb=C(fill)
    if line: r.line.color.rgb=C(line)
    else: r.line.fill.background()
    r.shadow.inherit=False; return r
def text(s,x,y,w,h,t,size=20,bold=False,color=INK,align=PP_ALIGN.LEFT,anchor=MSO_ANCHOR.TOP,font="Calibri",italic=False):
    tb=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=tb.text_frame; tf.word_wrap=True; tf.vertical_anchor=anchor
    tf.margin_left=tf.margin_right=Inches(0.05)
    lines=t if isinstance(t,list) else [t]
    for i,l in enumerate(lines):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.alignment=align
        r=p.add_run(); r.text=l.replace('**',''); r.font.size=Pt(size); r.font.bold=bold; r.font.color.rgb=C(color); r.font.name=font; r.font.italic=italic
    return tb
def bullets(s,x,y,w,h,items,size=22,color=INK,accent=IND):
    tb=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=tb.text_frame; tf.word_wrap=True
    for i,it in enumerate(items):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.space_after=Pt(10)
        r0=p.add_run(); r0.text="▪  "; r0.font.size=Pt(size); r0.font.color.rgb=C(accent); r0.font.bold=True; r0.font.name="Calibri"
        # **negrito**
        parts=re.split(r"(\*\*.+?\*\*)",it)
        for pt in parts:
            if not pt: continue
            r=p.add_run(); b=pt.startswith("**"); r.text=pt.strip("*") if b else pt; r.font.size=Pt(size); r.font.bold=b; r.font.color.rgb=C(accent if b else color); r.font.name="Calibri"
    return tb
def notes(s,t):
    if t: s.notes_slide.notes_text_frame.text=t
def base(prs,title,col,n,total,num):
    s=prs.slides.add_slide(prs.slide_layouts[6]); rect(s,0,0,W,0.14,col); rect(s,0,0.14,0.35,H-0.14,col)
    text(s,0.7,0.3,11.9,0.9,title,size=32,bold=True,color=INK,anchor=MSO_ANCHOR.MIDDLE)
    rect(s,0.75,1.2,1.4,0.06,AMB)
    text(s,0.7,7.0,8,0.35,f"MLOps · Capítulo {n}",size=11,color=GRY); text(s,11.6,7.0,1.4,0.35,str(num),size=11,color=GRY,align=PP_ALIGN.RIGHT)
    return s
def pic(s,fid,x,y,w,h):
    p=FIGS+fid+".png"; im=Image.open(p); iw,ih=im.size; r=min(w/iw,h/ih); pw,ph=iw*r,ih*r
    s.shapes.add_picture(p,Inches(x+(w-pw)/2),Inches(y+(h-ph)/2),Inches(pw),Inches(ph))

def parse_md(n):
    t=open(CAP+f"ch{n:02d}.md",encoding="utf-8").read()
    title=re.search(r"^# (.+)$",t,re.M).group(1); objs=re.findall(r"^@obj: (.+)$",t,re.M)
    mem=re.findall(r":::memorize O que levar deste capítulo\n(.*?)\n:::",t,re.S)
    mem=[l[2:].strip() for l in mem[0].split("\n") if l.startswith("- ")] if mem else []
    return title,[o.replace("*","") for o in objs],[m.replace("**","") for m in mem]

def build(n):
    title,objs,mem=parse_md(n); col=PARTCOL[n]; prs=Presentation(); prs.slide_width=Inches(W); prs.slide_height=Inches(H)
    num=[0]
    # capa
    s=prs.slides.add_slide(prs.slide_layouts[6]); rect(s,0,0,W,H,NAVY); num[0]+=1
    s.shapes.add_picture(FIGS+f"banner{n}.png",Inches(0),Inches(0.6),Inches(W)) 
    text(s,0.8,3.3,11.8,1.9,title,size=44,bold=True,color="FFFFFF")
    rect(s,0.85,5.35,2.0,0.08,AMB); text(s,0.8,5.6,11.8,0.5,"MLOps: do notebook à produção com Python",size=20,color="E5E7EB",italic=True); text(s,0.8,6.3,11.8,0.5,"Araken de Medeiros Santos",size=18,bold=True,color="FFFFFF")
    # objetivos
    num[0]+=1; s=base(prs,"O que você vai aprender",col,n,0,num[0]); 
    bullets(s,0.8,1.6,11.8,5.2,objs,size=26,accent=col); notes(s,"Leia os objetivos em voz alta e pergunte à turma o que já sabe sobre cada um.")
    for sp in CH[n]:
        num[0]+=1; k=sp[0]
        if k=="sec":
            _,t,bl,fig,nt=sp
            s=base(prs,t,col,n,0,num[0])
            if fig: bullets(s,0.8,1.6,6.2,5.3,bl,size=24,accent=col); pic(s,fig,7.2,1.6,5.7,5.0)
            else: bullets(s,0.8,1.6,11.8,5.3,bl,size=28,accent=col)
            notes(s,nt)
        elif k=="fig":
            _,t,fid,cap=sp; s=base(prs,t,col,n,0,num[0]); pic(s,fid,0.9,1.45,11.5,4.9); text(s,0.9,6.4,11.5,0.5,cap,size=15,color=GRY,align=PP_ALIGN.CENTER,italic=True)
        elif k=="ana":
            _,t,tx=sp; s=prs.slides.add_slide(prs.slide_layouts[6]); rect(s,0,0,W,H,"ECFEFF"); rect(s,0,0,0.5,H,CYA)
            text(s,1.0,0.7,11,0.6,"NO SEU DIA A DIA",size=16,bold=True,color=CYA); text(s,1.0,1.3,11.4,1.0,t,size=36,bold=True,color=INK)
            text(s,1.0,2.7,11.3,3.5,tx,size=28,color=INK); text(s,11.6,7.0,1.4,0.35,str(num[0]),size=11,color=GRY,align=PP_ALIGN.RIGHT)
        elif k=="imp":
            _,fr,sub=sp; s=prs.slides.add_slide(prs.slide_layouts[6]); rect(s,0,0,W,H,col); 
            text(s,1.0,1.6,11.3,3.0,fr,size=46,bold=True,color="FFFFFF",align=PP_ALIGN.CENTER,anchor=MSO_ANCHOR.MIDDLE); text(s,1.0,5.0,11.3,1.2,sub,size=22,color="FDE68A",align=PP_ALIGN.CENTER,italic=True)
        elif k=="code":
            _,t,code,cm=sp; s=base(prs,t,col,n,0,num[0]); rect(s,0.8,1.5,11.8,4.4,"0F172A",shape=MSO_SHAPE.ROUNDED_RECTANGLE)
            text(s,1.0,1.65,11.4,4.1,code.split("\n"),size=19,color="E2E8F0",font="Consolas"); text(s,0.8,6.1,11.8,0.8,cm,size=18,color=INK,bold=True)
        elif k=="quiz":
            _,q,a=sp; s=prs.slides.add_slide(prs.slide_layouts[6]); rect(s,0,0,W,H,"FFFBEB"); rect(s,0,0,0.5,H,AMB)
            text(s,1.0,0.7,11,0.6,"PARE E PENSE",size=16,bold=True,color=AMB); text(s,1.0,1.4,11.3,2.6,q,size=32,bold=True); rect(s,1.0,4.3,11.3,0.05,AMB)
            text(s,1.0,4.5,11.3,2.0,"Resposta: "+a,size=22,color=INK); notes(s,"Dê 1 minuto para a turma discutir em duplas antes de revelar a resposta."); 
        elif k=="cmp":
            _,t,hl,hr,L,R=sp; s=base(prs,t,col,n,0,num[0])
            for x,hh,items,cc in [(0.8,hl,L,col),(6.9,hr,R,AMB)]:
                rect(s,x,1.5,5.7,0.7,cc); text(s,x+0.1,1.5,5.5,0.7,hh,size=22,bold=True,color="FFFFFF",anchor=MSO_ANCHOR.MIDDLE)
                rect(s,x,2.2,5.7,4.5,"F8FAFC",line="E5E7EB"); bullets(s,x+0.15,2.35,5.4,4.3,items,size=19,accent=cc)
    # resumo
    num[0]+=1; s=prs.slides.add_slide(prs.slide_layouts[6]); rect(s,0,0,W,H,"FDF2F8"); rect(s,0,0,0.5,H,PNK)
    text(s,1.0,0.5,11,0.6,"MEMORIZE",size=16,bold=True,color=PNK); text(s,1.0,1.0,11.3,0.9,"O que levar deste capítulo",size=34,bold=True)
    bullets(s,1.0,2.0,11.5,5.0,mem,size=19,accent=PNK)
    # prática
    num[0]+=1; s=base(prs,"Hora de praticar",col,n,0,num[0])
    bullets(s,0.8,1.6,11.8,5,["**30 questões de múltipla escolha** e **20 discursivas** ao final do capítulo (gabarito no Apêndice A)","**Exercícios práticos em Python** com scikit-learn e as ferramentas do capítulo (soluções no Apêndice D)","**Projeto integrador do capítulo** (solução de referência no Apêndice E)","Dica de estudo: **rode, altere um parâmetro por vez e explique em voz alta o que mudou**"],size=22,accent=col)
    # numeração
    out=ROOT+f"/slides/Cap{n:02d}_{re.sub(r'[^A-Za-z0-9]+','_',unicodedata.normalize('NFD',title.split(':')[0]).encode('ascii','ignore').decode())[:40].strip('_')}.pptx"; prs.save(out); return out,len(prs.slides)
if __name__=="__main__":
    ns=[int(a) for a in sys.argv[1:]] or sorted(CH)
    for n in ns: print(build(n))

import re, json, sys, os, random
ROOT='/home/user/claude/livro-rl'
FIGS=ROOT+'/figs'
from PIL import Image

SUB=dict(zip('0123456789aehijklmnoprstuvx+-=()','₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ₊₋₌₍₎'))
def subs(t):
    def r1(m):
        x=m.group(2)
        return m.group(1)+''.join(SUB[c] for c in x) if all(c in SUB for c in x) else m.group(1)+'('+x+')'
    t=re.sub(r'(\w)_\(([^)]*)\)',r1,t)
    t=re.sub(r'(\w)_(\w)(?!\w)',lambda m:m.group(1)+(SUB.get(m.group(2),m.group(2))),t)
    return t
def clean(t):
    t=re.sub(r'!\[[^\]]*\]\([^)]*\)','',t)
    t=re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',t)
    t=t.replace('**','').replace('`','')
    t=re.sub(r'(?<![\w*])\*([^*\n]+)\*(?![\w*])',r'\1',t)
    t=subs(t)
    return re.sub(r'\s+',' ',t).strip()

def first_sentence(t,maxlen=150):
    t=clean(t)
    m=re.match(r'(.+?[.!?])(\s|$)',t)
    s=m.group(1) if m else t
    if len(s)>maxlen:
        cut=max(s.rfind(',',0,maxlen),s.rfind(' —',0,maxlen),s.rfind(';',0,maxlen))
        s=s[:cut] if cut>50 else s[:s.rfind(' ',0,maxlen)]
        s=s.rstrip(',;—- ')
    if s.count('(')>s.count(')'): s=s[:s.rfind('(')].rstrip(' ,;')
    s=s.rstrip(':')
    return s

def two_sentences(t,maxlen=220):
    t=clean(t)
    ss=re.findall(r'.+?[.!?](?:\s|$)',t)
    out=''
    for s in ss:
        if len((out+s).strip())>maxlen: break
        out+=s
    return out.strip() or first_sentence(t)

def blocks(lines):
    """yield (kind, data)"""
    i=0;n=len(lines);out=[]
    while i<n:
        l=lines[i]
        if not l.strip(): i+=1;continue
        if l.startswith('```'):
            j=i+1
            while j<n and not lines[j].startswith('```'): j+=1
            out.append(('code',lines[i+1:j]));i=j+1;continue
        if l.startswith(':::'):
            kind,_,title=l[3:].partition(' ')
            j=i+1;body=[]
            while j<n and lines[j].strip()!=':::': body.append(lines[j]);j+=1
            out.append(('callout',(kind,clean(title),body)));i=j+1;continue
        if l.startswith('$$'):
            f=l.strip().strip('$').strip()
            out.append(('formula',subs(f)));i+=1;continue
        m=re.match(r'!\[([^\]]+)\]\((.*)\)\s*$',l)
        if m: out.append(('fig',(m.group(2),m.group(1))));i+=1;continue
        if l.startswith('### '): out.append(('h3',clean(l[4:])));i+=1;continue
        if l.startswith('|'):
            j=i;rows=[]
            while j<n and lines[j].startswith('|'):
                rows.append([clean(c) for c in re.split(r'(?<!\\)\|',lines[j].strip().strip('|'))]);j+=1
            rows=[r for r in rows if not all(re.fullmatch(r'-*:?-*:?',c.replace(' ','')) for c in r)]
            out.append(('table',rows));i=j;continue
        if re.match(r'\s*([-*]|\d+\.)\s',l):
            items=[]
            while i<n and (re.match(r'\s*([-*]|\d+\.)\s',lines[i]) or (lines[i].startswith('  ') and lines[i].strip())):
                if re.match(r'\s*([-*]|\d+\.)\s',lines[i]): items.append(re.sub(r'^\s*([-*]|\d+\.)\s','',lines[i]))
                else: items[-1]+=' '+lines[i].strip()
                i+=1
            out.append(('list',items));continue
        j=i;p=[]
        while j<n and lines[j].strip() and not re.match(r'(```|:::|\$\$|!\[|###|\||\s*([-*]|\d+\.)\s)',lines[j]): p.append(lines[j]);j+=1
        out.append(('p',' '.join(p)));i=max(j,i+1)
    return out

def figinfo(fid):
    p=f'{FIGS}/{fid}.png'
    if not os.path.exists(p): return None
    w,h=Image.open(p).size
    return dict(path=p,w=w,h=h)

def chunk(items,k):
    import math
    n=len(items)
    if n==0: return []
    parts=math.ceil(n/k);sz=math.ceil(n/parts)
    return [items[i:i+sz] for i in range(0,n,sz)]

def quiz(num):
    txt=open(f'{ROOT}/cap/ex{num:02d}.txt').read().strip().split('\n\n')
    mc=[];dis=[]
    for b in txt:
        ls=b.strip().split('\n')
        if ls[0].startswith('MC:'):
            opts=[re.sub(r'^[a-d]\)\s*','',x) for x in ls[1:5]]
            r=ls[5][2:].strip();letter,_,ex=r.partition('|')
            mc.append(dict(q=ls[0][3:].strip(),opts=opts,ans=letter.strip(),exp=ex.strip()))
        elif ls[0].startswith('DISC:'):
            dis.append(ls[0][5:].strip())
    return mc,dis

def parse(num):
    src=open(f'{ROOT}/cap/ch{num:02d}.md').read().split('\n')
    title=next(l[2:] for l in src if l.startswith('# '))
    objs=[clean(l[5:]) for l in src if l.startswith('@obj:')]
    secs=[];cur=None
    for l in src:
        if l.startswith('## '):
            cur=dict(title=clean(l[3:]),lines=[]);secs.append(cur)
        elif cur is not None: cur['lines'].append(l)
    slides=[]
    slides.append(dict(t='title',num=num,title=title,banner=f'{FIGS}/banner{num}.png'))
    slides.append(dict(t='obj',items=objs))
    mapa=[re.sub(r'^\d+\.\d+\s*','',s['title']) for s in secs if not re.search(r'Exerc',s['title'])]
    slides.append(dict(t='roteiro',items=mapa[:12]))
    for s in secs:
        st=s['title']; short=re.sub(r'^\d+\.\d+\s*','',st)
        bl=blocks(s['lines'])
        if re.search(r'Exerc',st): continue
        notes=' '.join(clean(d) for k,d in bl if k=='p')[:1400]
        if re.search(r'Resumo do cap',st):
            items=[]
            for k,d in bl:
                if k=='list': items+= [first_sentence(x,170) for x in d]
                if k=='callout' and d[0]=='memorize':
                    items+= [first_sentence(x[2:],170) for x in d[2] if x.strip().startswith(('-','*'))]
            for ch in chunk(items,5): slides.append(dict(t='resumo',title='Resumo do capítulo',items=ch))
            continue
        if re.search(r'Mão na massa',st):
            subs=[d for k,d in bl if k=='h3']
            figs=[d for k,d in bl if k=='fig']
            slides.append(dict(t='section',title=short,sub='Hora de programar com Python'))
            slides.append(dict(t='cards',title='O que vamos programar',items=[re.sub(r'^\d+\.\d+\.\d+\s*','',x) for x in subs[:6]],notes='Execute os códigos do capítulo ao vivo ou peça que os alunos rodem em duplas.'))
            for fid,cap in [(a[1],a[0]) for a in [(f[0],f[1]) for f in figs]]:
                fi=figinfo(fid)
                if fi: slides.append(dict(t='fig',title=short,fig=fi,cap=first_sentence(cap.split('—',1)[-1],170),notes=cap))
            continue
        if re.search(r'Projeto do cap',st):
            slides.append(dict(t='section',title='Projeto do capítulo',sub=re.sub(r'^.*?:\s*','',short)))
            ps=[d for k,d in bl if k=='p']
            items=[first_sentence(p,170) for p in ps if len(clean(p))>40][:3]
            lst=[x for k,d in bl if k=='list' for x in d][:4]
            items+= [first_sentence(x,150) for x in lst if len(clean(x))>25]
            slides.append(dict(t='bullets',title='O desafio',items=items[:5],notes=notes))
            continue
        # pair formulas with their "Em palavras" paragraph
        nb=[];i=0
        while i<len(bl):
            k,d=bl[i]
            if k=='formula':
                emp=''
                if i+1<len(bl) and bl[i+1][0]=='p' and bl[i+1][1].lstrip('*').startswith('Em palavras'):
                    emp=clean(bl[i+1][1]);i+=1
                nb.append(('formula',(d,emp)))
            else: nb.append((k,d))
            i+=1
        bl=nb
        groups=[[None,[]]];extra=[]
        for idx,(k,d) in enumerate(bl):
            if k=='h3': groups.append([re.sub(r'^\d+\.\d+\.\d+\s*','',d),[]])
            elif k=='p':
                if d.rstrip().endswith(':') and idx+1<len(bl) and bl[idx+1][0]=='list':
                    groups[-1][1].append(('lead',d))
                elif d.rstrip().endswith(':'): pass
                else: groups[-1][1].append(('p',d))
            elif k=='list':
                groups[-1][1]+= [('li',x) for x in d]
            else: extra.append((k,d))
        bullets=[]
        for g,items in groups:
            if not items:
                continue
            if g:
                kind,first=items[0]
                if kind=='lead':
                    bullets.append(g+': '+first_sentence(first.rstrip(':')+'.',100).rstrip('.'))
                    for kk,x in items[1:4]: bullets.append('– '+first_sentence(x,120))
                else:
                    bullets.append(g+': '+first_sentence(first,140))
            else:
                n_p=0
                for kind,x in items:
                    if kind=='p' and n_p<3: bullets.append(first_sentence(x,150));n_p+=1
                    elif kind=='lead': bullets.append(first_sentence(x.rstrip(':')+'.',110).rstrip('.')+':')
                    elif kind=='li': bullets.append('– '+first_sentence(x,120))
        bullets=[b for b in bullets if len(b)>14]
        # group so that sub-items stay with their lead: chunk by 5
        txt=chunk(bullets[:15],5)
        for i,ch in enumerate(txt):
            slides.append(dict(t='bullets',title=short+(' (cont.)' if i else ''),items=ch,notes=notes if i==0 else '',num=st.split()[0]))
        for k,d in extra:
            if k=='fig':
                fi=figinfo(d[1])
                if fi: slides.append(dict(t='fig',title=short,fig=fi,cap=first_sentence(d[0].split('—',1)[-1],170),notes=clean(d[0])))
            elif k=='formula':
                f,emp=d
                if emp and len(f)<120: slides.append(dict(t='formula',title=short,formula=f,words=emp[:240]))
            elif k=='callout':
                kind,ct,body=d
                items=[clean(re.sub(r'^\s*[-*]\s','',x)) for x in body if x.strip()]
                if kind=='cotidiano': slides.append(dict(t='cotidiano',title=ct,items=[first_sentence(x,150) for x in items][:4],head='No dia a dia'))
                elif kind=='memorize':
                    if len(' '.join(items))<=260: slides.append(dict(t='quote',text=' '.join(items),title=ct))
                    else: slides.append(dict(t='cotidiano',title=ct,items=[first_sentence(x,140) for x in items][:6],head='Memorize'))
                elif kind=='exemplo': slides.append(dict(t='cotidiano',title=ct,items=[first_sentence(x,150) for x in items][:5],head='Exemplo'))
                elif kind=='atencao': slides.append(dict(t='cotidiano',title=ct,items=[first_sentence(x,150) for x in items][:4],head='Atenção'))
            elif k=='table':
                if 2<=len(d)<=7 and max(len(r) for r in d)<=4 and max(len(c) for r in d for c in r)<=48:
                    slides.append(dict(t='table',title=short,rows=d))
    mc,dis=quiz(num)
    slides.append(dict(t='section',title='Quiz relâmpago',sub='Teste o que você aprendeu'))
    pick=[mc[i] for i in (2,9,16,23)] if len(mc)>=24 else mc[:4]
    for i,q in enumerate(pick): slides.append(dict(t='quiz',n=i+1,**q))
    slides.append(dict(t='gabarito',items=[f"{i+1}) {q['ans']}  —  {first_sentence(q['exp'],110)}" for i,q in enumerate(pick)]))
    slides.append(dict(t='discuss',items=[first_sentence(d,190) for d in (dis[0],dis[7],dis[14])] if len(dis)>14 else dis[:3]))
    slides.append(dict(t='fim',title=title))
    return slides

if __name__=='__main__':
    for n in range(1,14):
        s=parse(n)
        json.dump(s,open(f'{ROOT}/slides/build/ch{n:02d}.json','w'),ensure_ascii=False,indent=1)
        print(n,len(s))

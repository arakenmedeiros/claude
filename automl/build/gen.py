"""Gera cap/chNN.md a partir de src/chNN.md.
Cada bloco python é executado. Marcadores na(s) primeira(s) linha(s) do bloco:
  #@saida    mostra a saída real logo após o bloco
  #@nb       bloco 'de notebook': executa precedido pelos blocos #@nb anteriores do capítulo (saída só do novo)
  #@naoexec  não executa (ferramenta indisponível neste ambiente); o texto deve avisar isso
"""
import re,sys,os,subprocess,glob,hashlib
os.environ["MPLBACKEND"]="Agg"
ROOT="/home/user/claude/automl"
CACHE=ROOT+"/build/.outcache"; os.makedirs(CACHE,exist_ok=True)
SEP="@@@SEP@@@"
def run(code,timeout=1500):
    h=hashlib.md5(code.encode()).hexdigest(); cf=f"{CACHE}/{h}.out"
    if os.path.exists(cf): return open(cf).read(),0
    p=f"/tmp/gen_{h}.py"; open(p,"w").write(code)
    r=subprocess.run(["python3","-W","ignore",p],capture_output=True,text=True,timeout=timeout,cwd="/tmp")
    if r.returncode!=0: return r.stderr[-1200:],1
    open(cf,"w").write(r.stdout); return r.stdout,0
def trunc(out,maxl=22,maxc=104):
    L=[l[:maxc-1]+"…" if len(l)>maxc else l for l in out.rstrip().split("\n")]
    if len(L)>maxl: L=L[:maxl]+["… (saída truncada)"]
    return "\n".join(L)
def flags_of(code):
    fl=set(); lines=code.split("\n")
    while lines and lines[0].startswith("#@"): fl.add(lines[0][2:].strip()); lines=lines[1:]
    return fl,"\n".join(lines)
bad=0
files=sorted(glob.glob(ROOT+"/src/ch*.md")) if len(sys.argv)<2 else [ROOT+f"/src/ch{int(a):02d}.md" for a in sys.argv[1:]]
for f in files:
    s=open(f).read(); out=[]; pos=0; n=0; prefix=""
    for m in re.finditer(r"```python\n(.*?)```\n",s,re.S):
        out.append(s[pos:m.start()]); pos=m.end(); n+=1
        fl,code=flags_of(m.group(1))
        if "naoexec" in fl:
            out.append("```python\n"+code+"```\n"); continue
        if "nb" in fl or "nbx" in fl:
            full=prefix+("print('%s')\n"%SEP if prefix else "")+code
            res,rc=run(full)
            if rc==0 and SEP in res: res=res.split(SEP+"\n",1)[1] if SEP+"\n" in res else res.split(SEP,1)[1]
            if "nb" in fl: prefix+=code+("\n" if not code.endswith("\n") else "")
        else:
            res,rc=run(code)
        if rc: bad+=1; print("FALHA",os.path.basename(f),"bloco",n); print(res)
        out.append("```python\n"+code+"```\n")
        if "saida" in fl and rc==0: out.append("```saída\n"+trunc(res)+"\n```\n")
    out.append(s[pos:]); open(ROOT+"/cap/"+os.path.basename(f),"w").write("".join(out)); print("ok",os.path.basename(f),n,"blocos")
print("falhas:",bad)

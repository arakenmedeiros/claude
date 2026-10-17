"""Gera cap/chNN.md a partir de src/chNN.md: executa blocos python marcados com '#@saida' e insere a saída real."""
import re,sys,os,subprocess,glob,hashlib
os.environ["MPLBACKEND"]="Agg"
ROOT="/home/user/claude/automl"
CACHE=ROOT+"/build/.outcache"; os.makedirs(CACHE,exist_ok=True)
def run(code):
    h=hashlib.md5(code.encode()).hexdigest(); cf=f"{CACHE}/{h}.out"
    if os.path.exists(cf): return open(cf).read(),0
    p=f"/tmp/gen_{h}.py"; open(p,"w").write(code)
    r=subprocess.run(["python3","-W","ignore",p],capture_output=True,text=True,timeout=900,cwd="/tmp")
    if r.returncode!=0: return r.stderr[-900:],1
    open(cf,"w").write(r.stdout); return r.stdout,0
def trunc(out,maxl=22,maxc=104):
    L=[l[:maxc-1]+"…" if len(l)>maxc else l for l in out.rstrip().split("\n")]
    if len(L)>maxl: L=L[:maxl]+["… (saída truncada)"]
    return "\n".join(L)
bad=0
files=sorted(glob.glob(ROOT+"/src/ch*.md")) if len(sys.argv)<2 else [ROOT+f"/src/ch{int(a):02d}.md" for a in sys.argv[1:]]
for f in files:
    s=open(f).read(); out=[]; pos=0; n=0
    for m in re.finditer(r"```python\n(.*?)```\n",s,re.S):
        out.append(s[pos:m.start()]); code=m.group(1); pos=m.end(); n+=1
        # todo bloco python é executado (garante que roda); só os marcados mostram saída
        marked=code.startswith("#@saida")
        run_code=code.replace("#@saida\n","",1) if marked else code
        if "#@naoexec" in code.split("\n")[0]:
            out.append("```python\n"+code.replace("#@naoexec\n","",1)+"```\n"); continue
        res,rc=run(run_code)
        if rc: bad+=1; print("FALHA",os.path.basename(f),"bloco",n); print(res)
        out.append("```python\n"+run_code+"```\n")
        if marked and rc==0: out.append("```saída\n"+trunc(res)+"\n```\n")
    out.append(s[pos:]); open(ROOT+"/cap/"+os.path.basename(f),"w").write("".join(out)); print("ok",os.path.basename(f),n,"blocos")
print("falhas:",bad)

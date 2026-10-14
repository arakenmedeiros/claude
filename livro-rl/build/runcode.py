"""Executa os blocos ```python de cada capitulo em sequencia (como celulas de um notebook).
Uso: python3 runcode.py cap/ch01.md [...]   (usa o shim de ucimlrepo para dados sinteticos)"""
import re,sys,subprocess,os,glob
root=os.path.dirname(os.path.abspath(__file__)); cap=os.path.join(root,"..","cap")
files=sys.argv[1:] or sorted(glob.glob(os.path.join(cap,"ch*.md"))+glob.glob(os.path.join(cap,"integrador.md"))+glob.glob(os.path.join(cap,"ap*.md")))
env=dict(os.environ,MPLBACKEND="Agg",PYTHONPATH=os.path.join(root,"shim")+":"+os.environ.get("PYTHONPATH",""),PYTHONWARNINGS="ignore")
for f in files:
    s=open(f).read()
    blocks=re.findall(r"```python\n(.*?)```",s,re.S)
    if not blocks: continue
    code="import warnings; warnings.filterwarnings('ignore')\n"+"\n".join(f"# --- bloco {i+1}\n{b}" for i,b in enumerate(blocks))
    p=f"/tmp/run_{os.path.basename(f)}.py"; open(p,"w").write(code)
    try:
        r=subprocess.run(["python3","-W","ignore",p],capture_output=True,text=True,timeout=1500,env=env,cwd="/tmp")
        print(("OK  " if r.returncode==0 else "FAIL"),os.path.basename(f),len(blocks),"blocos",flush=True)
        if r.returncode: print(r.stderr[-1500:])
        open(p+".out","w").write(r.stdout)
    except subprocess.TimeoutExpired: print("TIMEOUT",f)

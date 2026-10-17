import re,sys,os,subprocess,json,glob
os.environ["MPLBACKEND"]="Agg"
os.environ["PYTHONPATH"]="/home/user/claude/livro_af/cap/lib"
def parse(path):
    txt=open(path).read().replace("\r","")
    items=[];cur=None;mode=None
    for ln in txt.split("\n"):
        if ln.startswith("@@EX ") or ln.startswith("@@PROJ "):
            kind="EX" if ln.startswith("@@EX ") else "PROJ"
            head=ln.split(" ",1)[1]
            parts=[p.strip() for p in head.split("|")]
            if kind=="EX":
                num=int(parts[0]);title=parts[1];level=parts[2] if len(parts)>2 else ""
            else:
                title=parts[0];num=0;level=parts[1] if len(parts)>1 else ""
            cur={"kind":kind,"num":num,"title":title,"level":level,"stmt":[],"sol":[],"rubrica":[],"note":[]};items.append(cur);mode="stmt";continue
        if ln.startswith("@@ALEM"): cur=None; mode=None; continue
        if ln.startswith("@@SOL"): mode="sol";continue
        if ln.startswith("@@RUBRICA"): mode="rubrica";continue
        if ln.startswith("@@NOTE"): mode="note"; rest=ln[6:].strip(); 
        if cur is None: continue
        if mode=="note" and ln.startswith("@@NOTE"):
            if rest: cur["note"].append(rest)
            continue
        cur[mode].append(ln)
    return items
if __name__=="__main__":
    out={}
    files=sorted(glob.glob("/home/user/claude/livro_af/cap/pr/pr*.txt")) if len(sys.argv)<2 else sys.argv[1:]
    fails=0
    for f in files:
        ch=int(re.search(r"pr(\d+)",f).group(1))
        for it in parse(f):
            code="\n".join(it["sol"])
            m=re.findall(r"```python\n(.*?)```",code,re.S)
            if not m: print("SEM CODIGO",f,it["title"]); fails+=1; continue
            key=f"{ch}-{it['kind']}{it['num']}"
            p=f"/tmp/pr_{key}.py"; open(p,"w").write(m[0])
            try:
                r=subprocess.run(["python3","-W","ignore",p],capture_output=True,text=True,timeout=600,cwd="/tmp")
            except subprocess.TimeoutExpired:
                print("TIMEOUT",key); fails+=1; continue
            if r.returncode!=0:
                print("FAIL",key); print(r.stderr[-700:]); fails+=1
            else:
                print("OK  ",key,flush=True); out[key]=r.stdout
                if os.environ.get("SHOW"): print(r.stdout[-1500:])
    if len(sys.argv)<2: json.dump(out,open("/home/user/claude/livro_af/build/pr_out.json","w"),ensure_ascii=False,indent=1)
    print("falhas:",fails)

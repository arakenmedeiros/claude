import re,sys,subprocess,glob,os,tempfile
os.environ["MPLBACKEND"]="Agg"
files=sorted(glob.glob("../cap/ch*.md")) if len(sys.argv)<2 else sys.argv[1:]
for f in files:
    s=open(f).read()
    blocks=[]
    for m in re.findall(r"```python\n(.*?)```",s,re.S):
        if m.startswith("# continuação") and blocks: blocks[-1]+="\n"+m
        else: blocks.append(m)
    for i,m in enumerate(blocks):
        d=tempfile.mkdtemp(); p=os.path.join(d,"b.py"); open(p,"w").write(m)
        r=subprocess.run(["python3","-W","ignore","-I","b.py"],capture_output=True,text=True,cwd=d,timeout=900,env={**os.environ,"PYTHONPATH":""})
        tag="OK  " if r.returncode==0 else "FAIL"
        print(tag,os.path.basename(f),"bloco",i+1)
        if os.environ.get("SHOW") or r.returncode: print((r.stdout if r.returncode==0 else r.stderr[-800:]))

import re,sys,subprocess,glob,os,tempfile
os.environ["MPLBACKEND"]="Agg"
res=[]
files=sorted(glob.glob("cap/ch*.md")) if len(sys.argv)<2 else sys.argv[1:]
for f in files:
    s=open(f).read()
    blocks=re.findall(r"```python\n(.*?)```",s,re.S)
    for i,b in enumerate(blocks):
        p=f"/tmp/blk_{os.path.basename(f)}_{i}.py"
        open(p,"w").write(b)
        try:
            r=subprocess.run(["python3","-W","ignore",p],capture_output=True,text=True,timeout=300)
            ok=r.returncode==0
            print(("OK  " if ok else "FAIL"),f,i,flush=True)
            if not ok: print(r.stderr[-600:])
            else: open(p+".out","w").write(r.stdout)
        except subprocess.TimeoutExpired:
            print("TIMEOUT",f,i,flush=True)

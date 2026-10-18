import json,re,subprocess,sys
pdf=sys.argv[1]
H=json.load(open('headings.json'))
n=int(re.search(r"Pages:\s+(\d+)",subprocess.run(["pdfinfo",pdf],capture_output=True,text=True).stdout).group(1))
pages=[]
for p in range(1,n+1):
    t=subprocess.run(["pdftotext","-f",str(p),"-l",str(p),"-layout",pdf,"-"],capture_output=True,text=True).stdout
    pages.append(t)
def comp(s): return re.sub(r"\s+","",s)
# last TOC page: contains dot leaders
toc=[i for i,t in enumerate(pages) if t.count("......")>10]
start=max(toc)+1
buf="";offs=[]
for i in range(start,n):
    offs.append((len(buf),i+1)); buf+=comp(pages[i])
def page_of(pos):
    pg=None
    for o,p in offs:
        if o<=pos: pg=p
        else: break
    return pg
ptr=0;res={};miss=[]
for h in H:
    key=comp(h['text'])
    k=buf.find(key,ptr)
    if k<0: miss.append(h['text']); continue
    res[h['id']]=page_of(k); ptr=k+len(key)
json.dump(res,open('pages.json','w'))
print("toc pages",[t+1 for t in toc],"total",n,"headings",len(H),"found",len(res),"missing",miss[:10])
# monotonic check
prev=0;bad=0
for h in H:
    p=res.get(h['id'])
    if p is None: continue
    if p<prev: bad+=1; print("non-monotone",h['text'],p,prev)
    prev=p
print("bad",bad)

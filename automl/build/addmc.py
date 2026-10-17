import sys
def add(path, text):
    s=open(path).read().strip().split("\n\n")
    new=text.strip().split("\n\n")
    idx=max(i for i,b in enumerate(s) if b.startswith("MC:"))
    s[idx+1:idx+1]=new
    open(path,"w").write("\n\n".join(s)+"\n")

import re,glob
for f in sorted(glob.glob("/home/user/claude/automl/cap/ex*.txt")):
    s=open(f).read(); blocks=[b for b in re.split(r"\n\s*\n",s.strip()) if b.strip()]
    mc=[b for b in blocks if b.startswith("MC:")]; dc=[b for b in blocks if b.startswith("DISC:")]
    bad=0
    for b in mc:
        L=b.split("\n"); ok=len(L)==6 and all(L[i+1].startswith("abcd"[i]+") ") for i in range(4)) and re.match(r"R: [abcd] \| ",L[5])
        if not ok: bad+=1; print("  malformado:",L[0][:60])
    for b in dc:
        L=b.split("\n")
        if len(L)!=2 or not L[1].startswith("R: "): bad+=1; print("  malformado disc:",L[0][:60])
    print(f,len(mc),len(dc),"bad",bad)

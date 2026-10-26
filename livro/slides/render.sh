#!/bin/bash
# uso: render.sh arquivo.pptx pasta_saida [dpi]
SK=$(ls -d /root/.claude/skills/synced/*/pptx | head -1)
mkdir -p "$2"; rm -f "$2"/s-*.jpg "$2"/*.pdf
python3 $SK/scripts/office/soffice.py --headless --convert-to pdf --outdir "$2" "$1" >/dev/null 2>&1
pdftoppm -jpeg -r ${3:-60} "$2"/$(basename "${1%.pptx}").pdf "$2"/s
python3 - "$2" <<'PY'
import sys,glob
from PIL import Image
d=sys.argv[1]
fs=sorted(glob.glob(d+'/s-*.jpg')); ims=[Image.open(f) for f in fs]; w,h=ims[0].size
cols=4; per=12
for k in range(0,len(ims),per):
    sub=ims[k:k+per]; rows=(len(sub)+cols-1)//cols
    sheet=Image.new("RGB",(w*cols,h*rows),"white")
    for i,im in enumerate(sub): sheet.paste(im,((i%cols)*w,(i//cols)*h))
    sheet.save(f"{d}/grade_{k//per+1}.png")
print(len(ims),"slides;",(len(ims)+per-1)//per,"grades")
PY

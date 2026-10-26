#!/bin/bash
# Uso: bash qa.sh N  -> gera /tmp/qs_N/grid-*.jpg (grades 3x3 dos slides) para inspeção visual
cd "$(dirname "$0")"; export NODE_PATH=$PWD/node_modules
N=$1; nn=$(printf %02d $N); D=/tmp/qs_$nn; rm -rf $D; mkdir -p $D
node slides.js $N || exit 1
SK=$(ls -d /root/.claude/skills/synced/*/pptx | head -1)
python3 $SK/scripts/office/validate.py ../saida/pptx/Cap${nn}_slides.pptx | tail -1
python3 $SK/scripts/office/soffice.py --headless --convert-to pdf --outdir $D ../saida/pptx/Cap${nn}_slides.pptx >/dev/null 2>&1
pdftoppm -jpeg -r 50 $D/Cap${nn}_slides.pdf $D/s 2>/dev/null
python3 - <<PY
import glob
from PIL import Image
fs=sorted(glob.glob("$D/s-*.jpg"));
for g0 in range(0,len(fs),9):
    ims=[Image.open(f) for f in fs[g0:g0+9]];w,h=ims[0].size
    g=Image.new("RGB",(w*3,h*((len(ims)+2)//3)),"white")
    for i,im in enumerate(ims):g.paste(im,((i%3)*w,(i//3)*h))
    g.save("$D/grid-%d.jpg"%(g0//9+1))
print(len(fs),"slides; grades em $D/grid-*.jpg")
PY

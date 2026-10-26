#!/bin/bash
# uso: qa.sh N  -> renderiza, valida e gera grades em /tmp/qa
N=$(printf %02d $1)
export NODE_PATH=$(npm root -g):/home/user/claude/livro-rl/slides/build/node_modules
S=$(ls -d /root/.claude/skills/synced/*/pptx/scripts)
cd /home/user/claude/livro-rl && node slides/build/render.js $1 && python3 $S/office/validate.py slides/cap${N}_slides.pptx | tail -2
cd /tmp/qa && rm -f s-*.jpg grid*.png && python3 $S/office/soffice.py --headless --convert-to pdf /home/user/claude/livro-rl/slides/cap${N}_slides.pptx >/dev/null 2>&1 && pdftoppm -jpeg -r ${2:-45} cap${N}_slides.pdf s && python3 - <<'P'
import glob
from PIL import Image
fs=sorted(glob.glob('/tmp/qa/s-*.jpg'));ims=[Image.open(f) for f in fs];w,h=ims[0].size
cols=3
for k in range(0,len(ims),12):
    ch=ims[k:k+12];rows=(len(ch)+cols-1)//cols
    S=Image.new('RGB',(w*cols,h*rows),'white')
    for i,im in enumerate(ch):S.paste(im,((i%cols)*w,(i//cols)*h))
    S.save(f'/tmp/qa/grid{k//12}.png')
print(len(ims))
P

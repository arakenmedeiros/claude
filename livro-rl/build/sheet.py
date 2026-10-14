import sys
from PIL import Image
names=sys.argv[2:]; out=sys.argv[1]
ims=[Image.open(f"/home/user/claude/livro-rl/figs/{n}.png").convert("RGB") for n in names]
W=1100
ims=[im.resize((W,int(im.height*W/im.width))) for im in ims]
cols=2 if len(ims)>1 else 1
rows=[ims[i:i+cols] for i in range(0,len(ims),cols)]
H=sum(max(i.height for i in r) for r in rows); sheet=Image.new("RGB",(W*cols,H),"white"); y=0
for r in rows:
    for j,im in enumerate(r): sheet.paste(im,(j*W,y))
    y+=max(i.height for i in r)
sheet.save(out)

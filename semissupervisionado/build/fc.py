import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle, Polygon
import warnings
warnings.filterwarnings("ignore")

IND="#4F46E5"; CYA="#06B6D4"; AMB="#F59E0B"; RED="#EF4444"; GRN="#10B981"; VIO="#8B5CF6"; PNK="#EC4899"
INK="#1F2937"; GRY="#9CA3AF"; LIGHT="#F3F4F6"; MID="#E5E7EB"; NAVY="#111827"
CL=[IND,AMB,GRN,RED,CYA,VIO,PNK,"#84CC16"]
OUT="/home/user/claude/semissupervisionado/figs/"

plt.rcParams.update({
 "font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":MID,"axes.labelcolor":INK,
 "text.color":INK,"xtick.color":INK,"ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False,
 "axes.titleweight":"bold","axes.titlesize":10,"figure.dpi":100,"savefig.dpi":200,"axes.grid":False,
})

MAXW=8.6
def save(fig,name,tight=True):
    w,h=fig.get_size_inches()
    if w>MAXW:
        fig.set_size_inches(MAXW,h*MAXW/w)
        if len(fig.axes)>1:
            try: fig.tight_layout()
            except Exception: pass
    fig.savefig(OUT+name+".png",bbox_inches="tight" if tight else None,facecolor="white",pad_inches=0.12)
    plt.close(fig)
    print("fig",name,flush=True)

def clean(ax,box=False):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(box)
def canvas(w,h,xl=(0,100),yl=(0,100)):
    fig,ax=plt.subplots(figsize=(w,h)); ax.set_xlim(*xl); ax.set_ylim(*yl); ax.axis("off"); return fig,ax
def rbox(ax,x,y,w,h,text="",fc=LIGHT,ec=None,fs=9,tc=INK,bold=False,r=1.5,lw=1.2,ha="center",alpha=1):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec or fc,lw=lw,alpha=alpha,mutation_aspect=1)
    ax.add_patch(p)
    if text: ax.text(x+(w/2 if ha=="center" else 1.5),y+h/2,text,ha=ha,va="center",fontsize=fs,color=tc,fontweight="bold" if bold else "normal",linespacing=1.25)
    return p
def arrow(ax,p1,p2,c=INK,lw=1.6,style="-|>",rad=0.0,ms=12,ls="-"):
    a=FancyArrowPatch(p1,p2,arrowstyle=style,mutation_scale=ms,color=c,lw=lw,connectionstyle=f"arc3,rad={rad}",linestyle=ls)
    ax.add_patch(a); return a
def pts(ax,X,c=IND,s=14,**k): return ax.scatter(X[:,0],X[:,1],c=c,s=s,edgecolors="white",linewidths=0.3,**k)

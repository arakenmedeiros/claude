from fc import *
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Wedge

def phone(ax,x,y,s=1.0,c=IND):
    w,h=6*s,10*s
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle=f"round,pad=0,rounding_size={1.2*s}",fc="white",ec=c,lw=2,zorder=3))
    ax.add_patch(Rectangle((x-w/2+0.7*s,y-h/2+1.6*s),w-1.4*s,h-3.4*s,fc=c,alpha=0.18,lw=0,zorder=4))
    ax.add_patch(Circle((x,y-h/2+0.8*s),0.35*s,fc=c,lw=0,zorder=4))
def hospital(ax,x,y,s=1.0,c=RED):
    w,h=10*s,8*s
    ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,fc="white",ec=c,lw=2,zorder=3))
    ax.add_patch(Rectangle((x-0.9*s,y-1.0*s),1.8*s,4.4*s,fc=c,lw=0,zorder=4)); ax.add_patch(Rectangle((x-2.2*s,y+0.3*s),4.4*s,1.8*s,fc=c,lw=0,zorder=4))
    ax.add_patch(Rectangle((x-0.9*s,y-h/2),1.8*s,2.2*s,fc=c,alpha=0.5,lw=0,zorder=4))
def bank(ax,x,y,s=1.0,c=AMB):
    w,h=10*s,8*s
    ax.add_patch(Polygon([[x-w/2,y+h/2-3*s],[x,y+h/2],[x+w/2,y+h/2-3*s]],fc=c,ec=c,lw=1.5,zorder=3))
    for i in range(4): ax.add_patch(Rectangle((x-w/2+1*s+i*2.4*s,y-h/2+1.3*s),1.2*s,h-5.2*s,fc="white",ec=c,lw=1.5,zorder=4))
    ax.add_patch(Rectangle((x-w/2,y-h/2),w,1.1*s,fc=c,lw=0,zorder=3))
def server(ax,x,y,s=1.0,c=INK):
    w=10*s
    for i in range(3):
        yy=y-4.5*s+i*3.2*s
        ax.add_patch(FancyBboxPatch((x-w/2,yy),w,2.7*s,boxstyle=f"round,pad=0,rounding_size={0.5*s}",fc="white",ec=c,lw=2,zorder=3))
        ax.add_patch(Circle((x-w/2+1.4*s,yy+1.35*s),0.4*s,fc=GRN if i!=1 else AMB,lw=0,zorder=4))
        ax.plot([x-w/2+2.6*s,x+w/2-1*s],[yy+1.35*s]*2,c=MID,lw=2,zorder=4)
def cloud(ax,x,y,s=1.0,c=CYA):
    for dx,dy,r in [(-3,0,3),(0,1.5,3.6),(3.2,0,3),(0.2,-0.6,3)]:
        ax.add_patch(Circle((x+dx*s,y+dy*s),r*s,fc=c,alpha=0.25,ec=c,lw=1.5,zorder=2))
def db(ax,x,y,s=1.0,c=GRN):
    w,h=7*s,8*s
    ax.add_patch(Rectangle((x-w/2,y-h/2+1*s),w,h-2*s,fc="white",ec=c,lw=2,zorder=3))
    for yy in (y-h/2+1*s,y,y+h/2-1*s):
        ax.add_patch(Ellipse((x,yy),w,2*s,fc="white",ec=c,lw=2,zorder=4))
def lock(ax,x,y,s=1.0,c=GRN):
    ax.add_patch(Wedge((x,y+1.2*s),1.8*s,0,180,width=0.5*s,fc=c,lw=0,zorder=3))
    ax.add_patch(FancyBboxPatch((x-2.2*s,y-2*s),4.4*s,3.2*s,boxstyle=f"round,pad=0,rounding_size={0.5*s}",fc=c,ec=c,zorder=4))
    ax.add_patch(Circle((x,y-0.4*s),0.45*s,fc="white",lw=0,zorder=5))
def netglyph(ax,x,y,s=1.0,c=VIO):
    L=[[(-3,1.5),(-3,-1.5)],[(0,3),(0,0),(0,-3)],[(3,1.5),(3,-1.5)]]
    for a in L[0]:
        for b in L[1]: ax.plot([x+a[0]*s,x+b[0]*s],[y+a[1]*s,y+b[1]*s],c=c,lw=1,alpha=0.6,zorder=2)
    for a in L[1]:
        for b in L[2]: ax.plot([x+a[0]*s,x+b[0]*s],[y+a[1]*s,y+b[1]*s],c=c,lw=1,alpha=0.6,zorder=2)
    for l in L:
        for p in l: ax.add_patch(Circle((x+p[0]*s,y+p[1]*s),0.7*s,fc=c,ec="white",lw=0.8,zorder=3))
def person(ax,x,y,s=1.0,c=PNK):
    ax.add_patch(Circle((x,y+2*s),1.3*s,fc=c,lw=0,zorder=3)); ax.add_patch(Wedge((x,y-1.2*s),2.6*s,0,180,fc=c,lw=0,zorder=3))
def bug(ax,x,y,s=1.0,c=RED):
    ax.add_patch(Ellipse((x,y),3.4*s,4.4*s,fc=c,lw=0,zorder=4)); ax.add_patch(Circle((x,y+2.6*s),1.0*s,fc=c,lw=0,zorder=4))
    for k in (-1,0,1):
        ax.plot([x-1.7*s,x-3*s],[y+k*s,y+k*1.4*s],c=c,lw=1.5,zorder=3); ax.plot([x+1.7*s,x+3*s],[y+k*s,y+k*1.4*s],c=c,lw=1.5,zorder=3)

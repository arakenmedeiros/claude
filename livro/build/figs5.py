from fc import *
from matplotlib.colors import LinearSegmentedColormap
PART={1:(IND,CYA),2:(IND,CYA),3:(CYA,GRN),4:(CYA,GRN),5:(CYA,GRN),6:(CYA,GRN),7:(CYA,GRN),8:(GRN,AMB),9:(GRN,AMB),10:(AMB,RED),11:(AMB,RED),12:(RED,VIO),13:(VIO,PNK)}
def banner(n):
    c1,c2=PART[n]; rng=np.random.default_rng(100+n)
    fig,ax=plt.subplots(figsize=(10,1.9)); fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY); ax.set_xlim(0,10); ax.set_ylim(0,1.9); ax.axis("off")
    g=np.linspace(0,1,256)[None,:]; cm=LinearSegmentedColormap.from_list("b",[NAVY,"#1E1B4B",NAVY]); ax.imshow(g,extent=[0,10,0,1.9],aspect="auto",cmap=cm,zorder=0)
    centers=[(rng.uniform(4.2,9.5),rng.uniform(0.4,1.5)) for _ in range(4+n%3)]
    allp=[]
    for k,c in enumerate(centers):
        P=rng.normal(c,[0.35,0.22],(40,2)); P[:,1]=np.clip(P[:,1],0.05,1.85); col=c1 if k%2==0 else c2
        ax.scatter(P[:,0],P[:,1],s=rng.uniform(4,26,40),c=col,alpha=0.85,lw=0); ax.scatter(*c,s=1800,c=col,alpha=0.07,lw=0); allp.append(P)
    A=np.vstack(allp)
    for i in range(0,len(A),3):
        d=np.linalg.norm(A-A[i],axis=1); j=np.argsort(d)[1]; ax.plot([A[i,0],A[j,0]],[A[i,1],A[j,1]],c="white",alpha=0.12,lw=0.6)
    ax.text(0.35,0.95,f"{n:02d}",fontsize=62,fontweight="bold",color="white",alpha=0.92,va="center",ha="left")
    ax.text(0.35,0.12,"CAPÍTULO",fontsize=9,color=c2,fontweight="bold",va="bottom")
    fig.savefig(OUT+f"banner{n}.png",bbox_inches="tight",pad_inches=0,facecolor=NAVY,dpi=170); plt.close(fig); print("banner",n)

def cover():
    rng=np.random.default_rng(7); W,H=6.0,8.5
    fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off"); fig.patch.set_facecolor(NAVY)
    yy=np.linspace(0,1,512)[:,None]; cm=LinearSegmentedColormap.from_list("c",["#0B1020","#1E1B4B","#312E81","#0B1020"]); ax.imshow(yy*np.ones((1,2)),extent=[0,W,0,H],aspect="auto",cmap=cm,zorder=0,origin="lower")
    cen=[(1.4,5.6,IND),(4.4,6.4,CYA),(3.0,4.3,PNK),(1.2,3.8,AMB),(4.7,4.4,GRN),(2.4,6.9,VIO)]
    allp=[]
    for (x,y,c) in cen:
        n=70; P=rng.normal([x,y],[0.5,0.42],(n,2)); allp.append(P)
        ax.scatter(*(P.T),s=rng.uniform(6,60,n),c=c,alpha=0.9,lw=0,zorder=3)
        for r,a in [(1.1,0.05),(0.75,0.07),(0.45,0.10)]: ax.add_patch(Circle((x,y),r,fc=c,alpha=a,lw=0,zorder=1))
    A=np.vstack(allp)
    from sklearn.neighbors import NearestNeighbors
    nb=NearestNeighbors(n_neighbors=4).fit(A).kneighbors(A)[1]
    for i in range(len(A)):
        for j in nb[i,1:3]: ax.plot([A[i,0],A[j,0]],[A[i,1],A[j,1]],c="white",alpha=0.14,lw=0.5,zorder=2)
    ax.scatter([5.3,0.5,3.7],[7.6,5.2,3.1],s=[120,80,150],c=RED,marker="X",zorder=4); 
    for r in (0.9,1.6,2.3): ax.add_patch(Circle((3.0,5.4),r*2,fc="none",ec="white",alpha=0.04,lw=1))
    ax.text(0.45,2.55,"APRENDIZADO",fontsize=34,fontweight="bold",color="white",va="center",zorder=6)
    ax.text(0.45,1.85,"NÃO SUPERVISIONADO",fontsize=25.5,fontweight="bold",color=CYA,va="center",zorder=6)
    ax.plot([0.47,2.2],[1.45,1.45],c=AMB,lw=4,zorder=6)
    ax.text(0.47,1.12,"Do zero à prática com Python",fontsize=13.5,color="#E5E7EB",va="center",zorder=6,style="italic")
    ax.text(0.47,0.45,"Araken de Medeiros Santos",fontsize=15,fontweight="bold",color="white",va="center",zorder=6)
    ax.text(W-0.35,8.15,"2026",fontsize=10,color=GRY,ha="right",va="center",zorder=6)
    fig.savefig(OUT+"capa.png",facecolor=NAVY,dpi=220); plt.close(fig); print("capa")
if __name__=="__main__":
    cover()
    for n in range(1,14): banner(n)

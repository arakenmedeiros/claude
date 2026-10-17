from icons import *
import sys
sys.path.insert(0,"/home/user/claude/livro_af/cap/lib")
from compressao import *

def f5_assimetria():
    fig,ax=canvas(9,3.6)
    phone(ax,12,50,1.7,IND); server(ax,86,50,1.5)
    ax.text(12,26,"Cliente\n(celular)",ha="center",fontsize=9.5,fontweight="bold"); ax.text(86,24,"Servidor",ha="center",fontsize=9.5,fontweight="bold")
    arrow(ax,(78,60),(24,60),c=GRN,lw=6,ms=22)
    ax.text(51,67,"descida: modelo global",ha="center",fontsize=9.5,color=GRN,fontweight="bold"); ax.text(51,54.5,"rápida (ex.: 20 Mbit/s)",ha="center",fontsize=8.5,color=GRN)
    arrow(ax,(24,40),(78,40),c=RED,lw=2.2,ms=14)
    ax.text(51,31,"subida: atualização",ha="center",fontsize=9.5,color=RED,fontweight="bold"); ax.text(51,25,"lenta (ex.: 5 Mbit/s)",ha="center",fontsize=8.5,color=RED)
    save(fig,"f5_assimetria")

def f5_tempo():
    cen=[("Wi-Fi rápido\n(50/50 Mbit/s)",50,50),("Casa\n(20/5 Mbit/s)",20,5),("4G fraco\n(5/1 Mbit/s)",5,1)]
    MB=20; comp=30
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.6),sharey=True)
    for ax,(tit,fator) in zip(axs,[("Sem compressão",1),("Subida comprimida 8×",8)]):
        for i,(n,d,u) in enumerate(cen):
            td=MB*8/d; tu=MB*8/u/fator
            ax.bar(i,comp,color=IND,label="treino local" if i==0 else None); ax.bar(i,td,bottom=comp,color=GRN,label="descida" if i==0 else None); ax.bar(i,tu,bottom=comp+td,color=RED,label="subida" if i==0 else None)
            ax.text(i,comp+td+tu+4,f"{comp+td+tu:.0f} s",ha="center",fontsize=9,fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in cen],fontsize=8); ax.set_title(tit,fontsize=10.5); ax.set_ylim(0,260)
    axs[0].set_ylabel("tempo por rodada (s)"); axs[0].legend(frameon=False,fontsize=8.5,loc="upper left")
    save(fig,"f5_tempo")

def f5_quant():
    rng=np.random.default_rng(1); d=rng.normal(0,0.12,60)*np.hanning(60)**0.3; d=np.sort(d)[::1]; d=d[rng.permutation(60)]
    fig,axs=plt.subplots(1,4,figsize=(10,2.8),sharey=True)
    axs[0].bar(range(60),d,color=IND,width=0.85); axs[0].set_title("original (32 bits)",fontsize=9.5)
    for ax,b in zip(axs[1:],[4,2,1]):
        q=quantizar(d,b,np.random.default_rng(0)); ax.bar(range(60),q,color=AMB if b>1 else RED,width=0.85)
        ax.set_title(f"{b} bit{'s' if b>1 else ''} ({2**b} níveis)",fontsize=9.5)
    for ax in axs: ax.set_xticks([]); ax.axhline(0,c=GRY,lw=0.6)
    axs[0].set_ylabel("valor do delta")
    save(fig,"f5_quant")

def f5_topk():
    Xtr,Xte,ytr,yte=carregar_digits(); p=particao_iid(ytr,10,0)
    W=np.zeros((Xtr.shape[1],10)); Wc=treino_local(W,Xtr[p[0]],ytr[p[0]],5,0.1,32); d=(Wc-W).ravel()
    t=top_k(d,0.1)
    fig,axs=plt.subplots(2,1,figsize=(8.6,3.8),sharex=True,sharey=True)
    axs[0].bar(range(len(d)),d,color=IND,width=1.0); axs[0].set_title("atualização completa: 650 valores",fontsize=9.5,loc="left")
    cols=np.where(t!=0,AMB,"#E5E7EB"); axs[1].bar(range(len(d)),np.where(t!=0,d,0),color=AMB,width=1.0); axs[1].set_title("top-10%: 65 valores enviados (os demais viram zero)",fontsize=9.5,loc="left")
    axs[1].set_xlabel("índice do parâmetro")
    save(fig,"f5_topk")

def f5_tradeoff():
    Xtr,Xte,ytr,yte=carregar_digits()
    ms=[("float32",{}),("8 bits",dict(metodo="quantizar",param=8)),("4 bits",dict(metodo="quantizar",param=4)),("2 bits",dict(metodo="quantizar",param=2)),("1 bit",dict(metodo="quantizar",param=1)),
        ("top-10%",dict(metodo="topk",param=0.1)),("top-10% + EF",dict(metodo="topk",param=0.1,erro_acumulado=True)),("top-1%",dict(metodo="topk",param=0.01)),("top-1% + EF",dict(metodo="topk",param=0.01,erro_acumulado=True))]
    pts=[]
    for n,kw in ms:
        a=[];t=[]
        for s in range(3):
            p=particao_dirichlet(ytr,20,1.0,s); _,h,tt=fedavg_comprimido(p,Xtr,ytr,Xte,yte,seed=s,**kw); a.append(h[-1]); t.append(tt/1024)
        pts.append((n,np.mean(t),np.mean(a)))
    fig,ax=plt.subplots(figsize=(7,4.0))
    col={"float32":INK}
    for n,t,a in pts:
        c=INK if n=="float32" else (AMB if "bit" in n else (GRN if "EF" in n else RED))
        ax.scatter(t,a,s=90,c=c,ec="white",zorder=3,marker="o" if "bit" in n or n=="float32" else "s")
        off={"4 bits":(-4,-16),"top-10%":(8,-12),"top-10% + EF":(8,6),"8 bits":(2,-16),"2 bits":(-8,8)}.get(n,(5,6)); ax.annotate(n,(t,a),off,textcoords="offset points",fontsize=8.5)
    ax.set_xscale("log"); ax.set_xlabel("tráfego total de subida (KB, escala log)"); ax.set_ylabel("acurácia final"); ax.set_ylim(0.85,0.97)
    ax.set_title("Menos bytes, mesma acurácia: quantização e top-k com memória de erro",fontsize=9.5)
    save(fig,"f5_tradeoff")

if __name__=="__main__":
    for f in [f5_assimetria,f5_tempo,f5_quant,f5_topk,f5_tradeoff]: f()

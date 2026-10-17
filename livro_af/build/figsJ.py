from icons import *
import sys
sys.path.insert(0,"/home/user/claude/livro_af/cap/lib")
from robustez import *

def f9_taxonomia():
    fig,ax=canvas(10,4.4)
    rbox(ax,34,84,32,11,"Envenenamento em\naprendizado federado",fc=INK,tc="white",bold=True,fs=10)
    ax.text(25,70,"o que o atacante controla",ha="center",fontsize=9,color=GRY,style="italic"); ax.text(75,70,"o que o atacante quer",ha="center",fontsize=9,color=GRY,style="italic")
    arrow(ax,(42,84),(25,66),c=IND,lw=1.5,ms=9); arrow(ax,(58,84),(75,66),c=RED,lw=1.5,ms=9)
    boxes=[(2,"Dados","troca de rótulos,\nexemplos falsos",IND),(26,"Modelo","atualização arbitrária:\ninvertida, amplificada",CYA),(52,"Não direcionado","destruir a acurácia\nem geral",AMB),(76,"Direcionado\n(backdoor)","obedecer a um gatilho\nsecreto",RED)]
    for x,t,d,c in boxes:
        rbox(ax,x,38,21,16,t,fc=c,tc="white",bold=True,fs=9); rbox(ax,x,14,21,20,d,fc="white",ec=c,fs=8.2,lw=1.6)
    for x in (2+10.5,26+10.5): arrow(ax,(25,62),(x,54),c=INK,lw=0.8,ms=6)
    for x in (52+10.5,76+10.5): arrow(ax,(75,62),(x,54),c=INK,lw=0.8,ms=6)
    save(fig,"f9_taxonomia")

def f9_media_mediana():
    notas=np.array([7.0,6.5,7.5,8.0,7.0,6.0,7.2,7.8,6.8,7.1]); valores=[None,20,100,1000]
    fig,ax=plt.subplots(figsize=(7.4,3.8))
    xs=np.arange(4); med=[];mea=[];apa=[]
    for v in valores:
        a=notas if v is None else np.append(notas,v)
        mea.append(a.mean()); med.append(np.median(a)); s=np.sort(a); k=int(0.2*len(a)); apa.append(s[k:len(a)-k].mean())
    w=0.26
    ax.bar(xs-w,mea,w,color=RED,label="média"); ax.bar(xs,med,w,color=GRN,label="mediana"); ax.bar(xs+w,apa,w,color=IND,label="média aparada (20%)")
    ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["sem atacante","atacante\nenvia 20","atacante\nenvia 100","atacante\nenvia 1000"]); ax.set_ylabel("valor agregado (escala log)")
    for x,v in zip(xs-w,mea): ax.text(x,v*1.12,f"{v:.0f}" if v>20 else f"{v:.1f}",ha="center",fontsize=8,color=RED,fontweight="bold")
    ax.legend(frameon=False,fontsize=8.5,loc="upper left"); ax.set_title("Um único valor extremo arrasta a média, mas não a mediana",fontsize=10)
    save(fig,"f9_media_mediana")

def f9_ataques():
    Xtr,Xte,ytr,yte=carregar_digits(); K=20
    ags={"média":agg_media,"mediana":agg_mediana,"média\naparada":lambda D:agg_media_aparada(D,0.25),"Krum":lambda D:agg_krum(D,f=6,multi=1),"Multi-\nKrum":lambda D:agg_krum(D,f=6,multi=8)}
    atq=[("sinal","sinal invertido ×3"),("rotulos","troca de rótulos"),("ruido","ruído gaussiano")]
    M=np.zeros((3,5))
    for i,(a,_) in enumerate(atq):
        for j,(n,ag) in enumerate(ags.items()):
            M[i,j]=np.mean([fedavg_robusto(particao_iid(ytr,K,s),Xtr,ytr,Xte,yte,agregador=ag,maliciosos=range(6),ataque=a,seed=s)[1][-1] for s in range(3)])
    from matplotlib.colors import LinearSegmentedColormap
    cm=LinearSegmentedColormap.from_list("c",["#FCA5A5","#FEF3C7","#BBF7D0"])
    fig,ax=plt.subplots(figsize=(7.4,3.2)); ax.imshow(M,cmap=cm,vmin=0,vmax=1,aspect="auto")
    for i in range(3):
        for j in range(5): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=11,fontweight="bold",color=INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(list(ags.keys()),fontsize=9); ax.set_yticks(range(3)); ax.set_yticklabels([a[1] for a in atq],fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("Acurácia com 30% de clientes maliciosos (20 clientes)",fontsize=10)
    save(fig,"f9_ataques")

def f9_backdoor():
    Xtr,Xte,ytr,yte=carregar_digits(); K=20
    sel=np.where(yte==7)[0][:1]; x=Xte[sel]; xg=com_gatilho(x)
    def img(v): a=v[:, :-1].reshape(8,8); return a
    ags={"média":agg_media,"corte de\nnorma":lambda D:agg_corte_norma(D,0.5),"mediana":agg_mediana,"média\naparada":lambda D:agg_media_aparada(D,0.25)}
    asr=[]
    for n,ag in ags.items():
        asr.append(np.mean([fedavg_robusto(particao_iid(ytr,K,s),Xtr,ytr,Xte,yte,agregador=ag,maliciosos=range(2),ataque="backdoor",gama=1,seed=s)[2][-1] for s in range(3)]))
    fig=plt.figure(figsize=(9.4,3.4)); gs=fig.add_gridspec(1,3,width_ratios=[1,1,2.2])
    a1=fig.add_subplot(gs[0]); a1.imshow(img(x),cmap="gray_r"); a1.set_title("imagem normal",fontsize=9.5); a1.set_xticks([]); a1.set_yticks([])
    a2=fig.add_subplot(gs[1]); a2.imshow(img(xg),cmap="gray_r"); a2.set_title("com gatilho",fontsize=9.5); a2.set_xticks([]); a2.set_yticks([])
    a2.add_patch(plt.Rectangle((5.5,5.5),2,2,fill=False,ec=RED,lw=2))
    a3=fig.add_subplot(gs[2]); b=a3.bar(range(4),asr,color=[RED,AMB,GRN,IND],ec=INK,lw=0.6)
    for r,v in zip(b,asr): a3.text(r.get_x()+r.get_width()/2,v+0.02,f"{v:.2f}",ha="center",fontweight="bold",fontsize=9)
    a3.set_xticks(range(4)); a3.set_xticklabels(list(ags.keys()),fontsize=8.5); a3.set_ylim(0,1.1); a3.set_ylabel("taxa de sucesso do ataque (ASR)"); a3.set_title("2 de 20 clientes maliciosos (γ = 1)",fontsize=9.5)
    save(fig,"f9_backdoor")

def f9_noniid():
    Xtr,Xte,ytr,yte=carregar_digits()
    ags={"média":agg_media,"mediana":agg_mediana,"média\naparada":lambda D:agg_media_aparada(D,0.25),"Krum":lambda D:agg_krum(D,f=6,multi=1),"Multi-\nKrum":lambda D:agg_krum(D,f=6,multi=8)}
    alfas=[100,0.3,0.1]; M=np.zeros((3,5))
    for i,a in enumerate(alfas):
        for j,(n,ag) in enumerate(ags.items()):
            M[i,j]=np.mean([fedavg_robusto(particao_dirichlet(ytr,20,a,s),Xtr,ytr,Xte,yte,agregador=ag,seed=s)[1][-1] for s in range(3)])
    fig,ax=plt.subplots(figsize=(7.6,3.7)); w=0.26
    for i,(a,c) in enumerate(zip(alfas,[GRN,AMB,RED])):
        ax.bar(np.arange(5)+(i-1)*w,M[i],w,color=c,label=f"α = {a}"+(" (quase IID)" if a==100 else " (muito não-IID)" if a==0.1 else ""))
    ax.set_xticks(range(5)); ax.set_xticklabels(list(ags.keys()),fontsize=9); ax.set_ylabel("acurácia (sem atacantes)"); ax.set_ylim(0.3,1.02); ax.legend(frameon=False,fontsize=8.5,loc="lower left")
    ax.set_title("O preço da robustez: dados heterogêneos confundem os agregadores",fontsize=9.5)
    save(fig,"f9_noniid")

if __name__=="__main__":
    for f in [f9_taxonomia,f9_media_mediana,f9_ataques,f9_backdoor,f9_noniid]: f()

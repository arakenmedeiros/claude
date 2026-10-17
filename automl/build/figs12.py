from fc import *
import json
from scipy.ndimage import shift, rotate
from sklearn.datasets import load_digits

def f12_dominios():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"Três domínios fora da tabela comum",ha="center",fontsize=11,fontweight="bold")
    for i,(t,c) in enumerate([("Séries temporais",IND),("Textos",GRN),("Imagens",AMB)]):
        x=2+i*33; rbox(ax,x,10,30,78,"",fc="white",ec=c,lw=2.2,r=2); ax.text(x+15,80,t,ha="center",fontsize=9.5,fontweight="bold",color=c)
    # serie
    t=np.linspace(0,1,60); yv=40+22*t+7*np.sin(t*2*np.pi*5)
    ax.plot(4+t*26,yv-2,color=IND,lw=1.8); ax.text(17,22,"a ordem importa:\no hoje depende do ontem",ha="center",fontsize=8,color=INK)
    # texto
    for k,(w,c) in enumerate([("o produto",GRY),("não",RED),("é",GRY),("bom",GRN)]): rbox(ax,38+(k%2)*13,56-(k//2)*13,12,9,w,fc=c,ec=c,fs=7.8,tc="white",bold=True,r=1)
    ax.text(50,28,"palavras viram números\nantes do modelo",ha="center",fontsize=8,color=INK)
    # imagem
    d=load_digits().images[7]; ex=ax.inset_axes([0.70,0.40,0.26,0.30]); ex.imshow(d,cmap="Greys"); ex.axis("off")
    ax.text(83,28,"pixels vizinhos se\nrelacionam; a figura\npode girar ou deslocar",ha="center",fontsize=8,color=INK)
    save(fig,"f12_dominios")

def f12_validacao():
    fig,axs=plt.subplots(2,1,figsize=(7.4,3.6),sharex=True)
    N=24; rng=np.random.default_rng(3); idx=np.arange(N)
    va=set(rng.choice(N,6,replace=False).tolist())
    ax=axs[0]
    for i in idx: ax.add_patch(Rectangle((i,0),0.9,1,fc=(RED if i in va else MID),ec="white"))
    ax.set_xlim(0,N); ax.set_ylim(0,1); ax.set_yticks([]); ax.set_title("Validação aleatória: o modelo treina com o futuro (vazamento)",fontsize=9.5,color=RED,loc="left")
    ax=axs[1]
    for i in idx: ax.add_patch(Rectangle((i,0),0.9,1,fc=(IND if i<16 else (AMB if i<20 else MID)),ec="white"))
    ax.set_xlim(0,N); ax.set_ylim(0,1); ax.set_yticks([]); ax.set_title("Validação temporal: treina no passado, testa logo depois",fontsize=9.5,color=GRN,loc="left")
    ax.set_xlabel("tempo →  (azul = treino, laranja = validação, vermelho = validação sorteada)"); 
    for a in axs: a.spines["left"].set_visible(False)
    fig.tight_layout(); save(fig,"f12_validacao")

def f12_lags():
    fig,ax=canvas(8.6,3.6)
    ax.text(50,97,"A série vira tabela: atrasos (lags) + mês do ano",ha="center",fontsize=11,fontweight="bold")
    v=[315.9,316.5,317.1,317.8,318.3,318.5,317.9,316.2]
    for i,x in enumerate(v):
        rbox(ax,3+i*6.2,62,5.6,12,f"{x:.1f}",fc=(IND if i<3 else (GRN if i==3 else LIGHT)),ec=(IND if i<3 else (GRN if i==3 else MID)),fs=6.4,tc=("white" if i<4 else INK),r=1)
    ax.text(18,80,"valores passados (L = 3)",ha="center",fontsize=8,color=IND); ax.text(24.5,80,"alvo",ha="center",fontsize=8,color=GRN)
    arrow(ax,(18,58),(18,46),lw=1.8,ms=10); arrow(ax,(24.5,60),(70,46),lw=1.8,ms=10)
    rbox(ax,6,22,50,22,"",fc="white",ec=IND,lw=2,r=2)
    ax.text(31,38,"atributos: y(t−1), y(t−2), y(t−3), mês do ano",ha="center",fontsize=8.4,color=INK); ax.text(31,28,"→ qualquer modelo de tabela (Ridge, boosting...)",ha="center",fontsize=8,color=GRY)
    rbox(ax,60,22,36,22,"alvo: y(t)",fc=GRN,ec=GRN,fs=9,tc="white",bold=True,r=2)
    ax.text(50,10,"Com diferenciação, o alvo é y(t) − y(t−12): quanto o valor mudou em relação ao mesmo mês do ano anterior.",ha="center",fontsize=7.8,color=GRY,style="italic")
    save(fig,"f12_lags")

def f12_previsao():
    d=json.load(open("/home/user/claude/automl/build/f12_ts.json")); h=len(d["hist"]); H=len(d["real"])
    fig,ax=plt.subplots(figsize=(7.0,3.5)); x0=np.arange(h); x1=np.arange(h,h+H)
    ax.plot(x0,d["hist"],color=GRY,lw=1.8,label="histórico (treino)"); ax.plot(x1,d["real"],color=INK,lw=2.4,label="real")
    ax.plot(x1,d["auto"],color=GRN,lw=2,label="mini-AutoML (Ridge)"); ax.plot(x1,d["ing"],color=AMB,lw=1.8,ls="--",label="ingênua")
    ax.plot(x1,d["pior"],color=RED,lw=1.8,ls=":",label="pior configuração")
    ax.axvline(h-0.5,color=MID,lw=1); ax.set_xlabel("meses (últimos 72 da série)"); ax.set_ylabel("CO₂ (ppm)")
    ax.set_title("Previsão um mês à frente nos últimos 36 meses",fontsize=10); ax.legend(frameon=False,fontsize=7.5,loc="upper left"); ax.grid(color=MID,lw=.5)
    save(fig,"f12_previsao")

def f12_texto():
    fig,ax=canvas(8.6,3.4)
    ax.text(50,96,"O pipeline de texto",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,40,19,36,"\"o produto\nnão é bom\"",fc=GRY,ec=GRY,fs=8.4,tc="white",bold=True,r=2)
    rbox(ax,28,40,22,36,"Vetorização\nTF-IDF\n(n-gramas)",fc=IND,ec=IND,fs=8.6,tc="white",bold=True,r=2)
    rbox(ax,57,40,18,36,"vetor\n[0,0; 0,6;\n0,0; 0,4 ...]",fc="#EEF2FF",ec=IND,fs=7.8,tc=INK,lw=2,r=2)
    rbox(ax,81,40,17,36,"Classificador\n→ negativo",fc=GRN,ec=GRN,fs=8.4,tc="white",bold=True,r=2)
    for x in (21.5,50.5,75.5): arrow(ax,(x,58),(x+5.5,58),lw=2,ms=11)
    ax.text(50,22,"Escolhas do AutoML: n-gramas (1,1) ou (1,2)? min_df? sublinear_tf? qual classificador?",ha="center",fontsize=8.3,color=INK)
    ax.text(50,10,"Um bigrama como \"não bom\" recupera o sentido que as palavras soltas perdem.",ha="center",fontsize=8.3,color=GRY,style="italic")
    save(fig,"f12_texto")

def f12_augmentation():
    X=load_digits().images[7]; r=np.random.default_rng(2)
    fig,axs=plt.subplots(1,6,figsize=(8.0,1.9)); tit=["original","deslocado","girado","com ruído","combinado 1","combinado 2"]
    ims=[X,shift(X,(1,-1),order=1),rotate(X,15,reshape=False,order=1),X+r.normal(0,1.5,X.shape),rotate(shift(X,(-1,1),order=1),-10,reshape=False,order=1)+r.normal(0,.8,X.shape),rotate(shift(X,(1,1),order=1),8,reshape=False,order=1)]
    for a,im,t in zip(axs,ims,tit): a.imshow(im,cmap="Greys"); a.axis("off"); a.set_title(t,fontsize=8)
    fig.suptitle("Um dígito, vários exemplos de treino",fontsize=9.6,fontweight="bold",y=1.04); fig.tight_layout(); save(fig,"f12_augmentation")

def f12_resultado_aug():
    L=["sem ampliação","melhor política"]; limpo=[0.962,0.939]; dist=[0.529,0.873]
    fig,ax=plt.subplots(figsize=(5.8,3.3)); x=np.arange(2); w=.34
    b1=ax.bar(x-w/2,limpo,w,color=IND,label="teste limpo"); b2=ax.bar(x+w/2,dist,w,color=AMB,label="teste deslocado/girado")
    for b in list(b1)+list(b2): ax.text(b.get_x()+b.get_width()/2,b.get_height()+.012,f"{b.get_height():.3f}",ha="center",fontsize=8.5,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(L); ax.set_ylim(0,1.12); ax.set_ylabel("acurácia"); ax.legend(frameon=False,fontsize=8,loc="upper center",ncol=2)
    ax.set_title("Mini-AutoAugment no Digits (300 imagens de treino)",fontsize=10); save(fig,"f12_resultado_aug")

def f12_quadro():
    fig,ax=canvas(8.6,3.8)
    ax.text(50,97,"O que muda e o que permanece",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,52,96,34,"",fc="#EEF2FF",ec=IND,lw=2,r=2)
    ax.text(50,76,"O que PERMANECE",ha="center",fontsize=9.5,fontweight="bold",color=IND)
    ax.text(50,60,"espaço de busca  •  avaliação honesta (validação + teste intocado)  •  linhas de base  •  orçamento",ha="center",fontsize=8.6,color=INK)
    for i,(t,d,c) in enumerate([("Séries","validação temporal,\nlags, diferenciação",IND),("Texto","representação (n-gramas)\nou modelo pré-treinado",GRN),("Imagens","ampliação de dados,\ntransferência",AMB)]):
        x=2+i*33; rbox(ax,x,6,30,40,"",fc="white",ec=c,lw=2.2,r=2); ax.text(x+15,38,t,ha="center",fontsize=9.5,fontweight="bold",color=c); ax.text(x+15,22,d,ha="center",va="center",fontsize=8.4,color=INK)
    ax.text(50,49,"O que MUDA em cada domínio:",ha="center",va="center",fontsize=8,fontweight="bold",color=GRY)
    save(fig,"f12_quadro")

if __name__=="__main__":
    for fn_ in (f12_dominios,f12_validacao,f12_lags,f12_previsao,f12_texto,f12_augmentation,f12_resultado_aug,f12_quadro):
        try: fn_()
        except FileNotFoundError as e: print("pendente",fn_.__name__,e)

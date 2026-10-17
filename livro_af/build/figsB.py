from icons import *
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def _loss(w1,w2): return 0.5*(w1-1.0)**2*1.0+ 0.5*(w2+0.5)**2*6.0 + 0.3*w1*w2
def _grad(w): return np.array([(w[0]-1.0)+0.3*w[1], 6.0*(w[1]+0.5)+0.3*w[0]])

def f2_montanha():
    fig,ax=plt.subplots(figsize=(6.4,4.4))
    g1,g2=np.meshgrid(np.linspace(-2.5,3.5,200),np.linspace(-2.5,2,200))
    Z=_loss(g1,g2); cs=ax.contourf(g1,g2,Z,levels=18,cmap="Blues_r",alpha=0.9); ax.contour(g1,g2,Z,levels=18,colors="white",linewidths=0.4)
    w=np.array([-2.0,1.6]); path=[w.copy()]
    for _ in range(25): w=w-0.12*_grad(w); path.append(w.copy())
    P=np.array(path); ax.plot(P[:,0],P[:,1],"o-",c=AMB,ms=4,lw=1.8,mec="white")
    ax.scatter(*P[0],s=140,c=RED,zorder=5,ec="white"); ax.text(P[0,0]+0.15,P[0,1]+0.1,"início\n(palpite)",fontsize=8.5,color=INK,fontweight="bold")
    ax.scatter(*P[-1],s=200,c=GRN,marker="*",zorder=5,ec="white"); ax.text(P[-1,0]+0.35,P[-1,1]-0.75,"vale\n(menor erro)",fontsize=8.5,color=INK,fontweight="bold")
    ax.set_xlabel("parâmetro w₁"); ax.set_ylabel("parâmetro w₂"); ax.set_title("Descendo a montanha do erro com passos de tamanho η")
    save(fig,"f2_montanha")

def f2_lr():
    fig,axs=plt.subplots(1,3,figsize=(9.5,3.0),sharey=True)
    x=np.linspace(-3.2,3.2,200); 
    for ax,(eta,titulo,cor) in zip(axs,[(0.05,"η pequeno demais: lento",AMB),(0.45,"η adequado: converge",GRN),(1.05,"η grande demais: diverge",RED)]):
        ax.plot(x,x**2,c=INK,lw=2); w=-3.0; xs=[w]
        for _ in range(8): w=w-eta*2*w; xs.append(w)
        xs=np.array(xs); xs=np.clip(xs,-3.2,3.2)
        ax.plot(xs,xs**2,"o-",c=cor,ms=5,lw=1.5,mec="white"); ax.set_title(titulo,fontsize=9.5,color=cor); ax.set_xlabel("w"); ax.set_ylim(-0.5,11)
    axs[0].set_ylabel("erro")
    save(fig,"f2_lr")

def f2_sgd():
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.6),sharex=True,sharey=True)
    g1,g2=np.meshgrid(np.linspace(-2.5,3.5,150),np.linspace(-2.5,2,150)); Z=_loss(g1,g2)
    rng=np.random.default_rng(3)
    for ax,(nome,ruido) in zip(axs,[("Gradiente com todos os dados",0.0),("Gradiente com mini-lotes (SGD)",1.6)]):
        ax.contourf(g1,g2,Z,levels=16,cmap="Blues_r",alpha=0.85); ax.contour(g1,g2,Z,levels=16,colors="white",linewidths=0.3)
        w=np.array([-2.0,1.6]); P=[w.copy()]
        for _ in range(45): w=w-0.08*(_grad(w)+rng.normal(0,ruido,2)); P.append(w.copy())
        P=np.array(P); ax.plot(P[:,0],P[:,1],"-",c=AMB,lw=1.6); ax.scatter(P[::3,0],P[::3,1],s=14,c=AMB,ec="white",lw=0.5,zorder=4)
        ax.scatter(1.1,-0.7,s=170,c=GRN,marker="*",ec="white",zorder=5); ax.set_title(nome,fontsize=10); ax.set_xlabel("w₁")
    axs[0].set_ylabel("w₂")
    save(fig,"f2_sgd")

def f2_overfit():
    X,y=load_digits(return_X_y=True); Xtr,Xva,ytr,yva=train_test_split(X,y,train_size=200,test_size=600,random_state=0,stratify=y)
    esc=StandardScaler().fit(Xtr); Xtr,Xva=esc.transform(Xtr),esc.transform(Xva)
    m=MLPClassifier(hidden_layer_sizes=(128,),solver="sgd",learning_rate_init=0.05,batch_size=20,random_state=0)
    tr,va,ep=[],[],[]
    for e in range(150):
        m.partial_fit(Xtr,ytr,classes=np.arange(10)); tr.append(1-m.score(Xtr,ytr)); va.append(1-m.score(Xva,yva))
    fig,ax=plt.subplots(figsize=(6.6,3.6))
    ax.plot(tr,c=IND,lw=2,label="erro no treino"); ax.plot(va,c=RED,lw=2,label="erro na validação")
    k=int(np.argmin(va)); ax.axvline(k,c=GRY,ls="--"); ax.text(k+2,max(va)*0.8,"validação para de\nmelhorar aqui",fontsize=8.5,color=INK)
    ax.set_xlabel("épocas de treino"); ax.set_ylabel("taxa de erro"); ax.legend(frameon=False); ax.set_title("A lacuna: o erro de treino vai a zero, o de validação fica acima",fontsize=9.5)
    save(fig,"f2_overfit")

def f2_scaler():
    fig,ax=canvas(10,4.0)
    for i,(c,ic,y) in enumerate([(IND,phone,78),(RED,hospital,50),(AMB,bank,22)]):
        ic(ax,10,y,1.0,c)
        rbox(ax,20,y-7,32,14,f"cliente {i+1} envia só:\nn, Σx e Σx²",fc="white",ec=c,fs=8.5,lw=1.8)
        arrow(ax,(52,y),(64,50+(y-50)*0.3),c=c,lw=2)
    server(ax,76,50,1.1)
    rbox(ax,62,8,36,18,"servidor calcula\nmédia global μ e\ndesvio global σ",fc="#ECFDF5",ec=GRN,fs=9,bold=True,lw=1.8)
    ax.text(76,34,"Servidor",ha="center",fontsize=9.5,fontweight="bold")
    ax.text(30,96,"Nenhum dado bruto sai do cliente: só contagens e somas",ha="center",fontsize=10,color=GRN,fontweight="bold")
    save(fig,"f2_scaler")

def f2_particao():
    fig,ax=canvas(10,2.8)
    rbox(ax,2,52,60,28,"Treino (60%): ajusta os parâmetros",fc=IND,tc="white",bold=True,fs=10)
    rbox(ax,62,52,20,28,"Validação\n(20%)",fc=AMB,tc="white",bold=True,fs=10)
    rbox(ax,82,52,16,28,"Teste\n(20%)",fc=RED,tc="white",bold=True,fs=10)
    ax.text(32,36,"o modelo aprende aqui",ha="center",fontsize=9,color=INK); ax.text(72,36,"escolhe hiperparâmetros",ha="center",fontsize=9,color=INK); ax.text(90,36,"nota final,\nuso único",ha="center",fontsize=9,color=INK)
    ax.text(50,10,"Em aprendizado federado, cada cliente também precisa ter seu treino, validação e teste locais.",ha="center",fontsize=9.5,style="italic",color=GRY)
    save(fig,"f2_particao")

if __name__=="__main__":
    for f in [f2_montanha,f2_lr,f2_sgd,f2_overfit,f2_scaler,f2_particao]: f()

from icons import *
import sys
sys.path.insert(0,"/home/user/claude/livro_af/cap/lib")
from privacidade import *

def f8_camadas():
    fig,ax=canvas(10,4.4)
    cols=[IND,CYA,GRN]
    ax.text(50,95,"cada camada protege um trecho diferente do caminho",ha="center",fontsize=9.5,style="italic",color=GRY)
    phone(ax,8,55,1.3,IND); server(ax,92,55,1.2)
    ax.text(8,36,"Cliente",ha="center",fontsize=9,fontweight="bold"); ax.text(92,34,"Servidor",ha="center",fontsize=9,fontweight="bold")
    arrow(ax,(16,55),(82,55),c=INK,lw=2.4)
    lab=[("Privacidade\ndiferencial",IND,"ruído no que sai\n(protege o\nresultado)",28),("Agregação\nsegura",CYA,"máscaras que\nse cancelam\n(servidor só vê\na soma)",50),("Criptografia\nhomomórfica\ne SMPC",GRN,"cálculo sobre\ndados cifrados\nou repartidos",72)]
    for t,c,d,x in lab:
        rbox(ax,x-10,64,20,20,t,fc=c,tc="white",bold=True,fs=8.2); ax.plot([x,x],[66,55],c=c,lw=2)
        ax.add_patch(Circle((x,55),2.2,fc="white",ec=c,lw=3,zorder=4))
        rbox(ax,x-10,12,20,24,d,fc="white",ec=c,fs=7.8,lw=1.6); ax.plot([x,x],[55,36],c=c,lw=1.3,ls=":")
    save(fig,"f8_camadas")

def f8_dp():
    x=np.linspace(-9,12,500); g=lambda m,s:np.exp(-(x-m)**2/(2*s*s))/(s*np.sqrt(2*np.pi))
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.2),sharey=False)
    for ax,(s,t,c) in zip(axs,[(0.8,"pouco ruído: distinguível (ε grande)",RED),(3.0,"muito ruído: quase igual (ε pequeno)",GRN)]):
        ax.fill_between(x,g(0,s),color=IND,alpha=0.35,label="com os dados da pessoa"); ax.fill_between(x,g(3,s),color=AMB,alpha=0.35,label="sem os dados da pessoa")
        ax.plot(x,g(0,s),c=IND,lw=2); ax.plot(x,g(3,s),c=AMB,lw=2); ax.set_title(t,fontsize=9.5,color=c); ax.set_yticks([]); ax.set_xlabel("saída do mecanismo")
    axs[1].legend(frameon=False,fontsize=8,loc="upper right")
    save(fig,"f8_dp")

def f8_dpfedavg():
    fig,ax=canvas(10,4.0)
    cl=[(10,78,IND,phone),(10,50,RED,hospital),(10,22,AMB,bank)]
    for i,(x,y,c,f) in enumerate(cl):
        f(ax,x,y,0.8,c); rbox(ax,22,y-6,22,12,f"Δ_{i+1}  →  corte\nnorma ≤ S",fc="white",ec=c,fs=8.5,lw=1.6); arrow(ax,(15,y),(22,y),c=c,lw=1.3,ms=8); arrow(ax,(44,y),(54,50+(y-50)*0.3),c=c,lw=1.6)
    rbox(ax,54,38,16,24,"Σ  atualizações\ncortadas",fc="#EEF2FF",ec=IND,fs=9,bold=True,lw=1.8)
    ax.text(62,28,"+",fontsize=22,ha="center",fontweight="bold",color=RED)
    rbox(ax,54,10,16,14,"ruído\nN(0, (z·S)²)",fc="#FEF2F2",ec=RED,fs=8.5,lw=1.6)
    arrow(ax,(70,50),(80,50),c=INK,lw=2); rbox(ax,80,38,18,24,"÷ m\nnovo modelo\nglobal",fc="#ECFDF5",ec=GRN,fs=9,bold=True,lw=1.8)
    save(fig,"f8_dpfedavg")

def f8_tradeoff():
    Xtr,Xte,ytr,yte=carregar_digits()
    zs=[0.3,0.5,1,2,3,5]
    fig,ax=plt.subplots(figsize=(7,4.0))
    for K,c in ((20,RED),(100,IND)):
        base=np.mean([fedavg_dp(particao_iid(ytr,K,s),Xtr,ytr,Xte,yte,z=0,S=0.5,seed=s)[1][-1] for s in range(3)])
        accs=[np.mean([fedavg_dp(particao_iid(ytr,K,s),Xtr,ytr,Xte,yte,z=z,S=0.5,seed=s)[1][-1] for s in range(3)]) for z in zs]
        eps=[epsilon_rdp(z,30) for z in zs]
        ax.plot(eps,accs,"o-",c=c,lw=2,label=f"{K} clientes"); ax.axhline(base,c=c,ls=":",lw=1.2)
        ax.annotate(f"sem ruído ({K} clientes)",xy=(0.99,base+0.003),xycoords=ax.get_yaxis_transform(),color=c,fontsize=8,ha="right",va="bottom")
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel("ε (escala log; mais privado → direita)"); ax.set_ylabel("acurácia final"); ax.legend(frameon=False,loc="lower left",fontsize=9)
    ax.set_title("Privacidade × utilidade no DP-FedAvg (30 rodadas, δ = 10⁻⁵)",fontsize=9.5)
    save(fig,"f8_tradeoff")

def f8_secagg():
    fig,ax=canvas(10,4.6)
    cols=[IND,RED,AMB]; xs=[16,50,84]
    for i,(x,c) in enumerate(zip(xs,cols)):
        phone(ax,x,78,1.0,c); ax.text(x,69,f"u_{i+1}",ha="center",fontsize=10,fontweight="bold",color=c)
    # segredos
    for (a,b,t,yy) in [(0,1,"s₁₂",90),(1,2,"s₂₃",90),(0,2,"s₁₃",97)]:
        ax.plot([xs[a],xs[b]],[yy,yy],c=GRN,lw=1.5,ls=":")
        ax.text((xs[a]+xs[b])/2,yy+1.2,t+" (segredo do par)",ha="center",fontsize=7.8,color=GRN,fontweight="bold")
    msgs=["u₁ + s₁₂ + s₁₃","u₂ − s₁₂ + s₂₃","u₃ − s₁₃ − s₂₃"]
    for x,c,m in zip(xs,cols,msgs):
        rbox(ax,x-14,46,28,12,m,fc="white",ec=c,fs=9.5,bold=True,lw=2); arrow(ax,(x,65),(x,58.5),c=c,lw=1.5,ms=9)
        arrow(ax,(x,46),(50,31),c=c,lw=1.6,ms=10)
    server(ax,50,20,0.8)
    rbox(ax,66,10,32,20,"servidor soma tudo:\nu₁ + u₂ + u₃\n(as máscaras se cancelam)",fc="#ECFDF5",ec=GRN,fs=9.2,bold=True,lw=1.8)
    ax.text(14,22,"cada mensagem sozinha\nparece ruído aleatório",fontsize=8.8,color=RED,fontweight="bold",va="center")
    save(fig,"f8_secagg")

if __name__=="__main__":
    for f in [f8_camadas,f8_dp,f8_dpfedavg,f8_tradeoff,f8_secagg]: f()

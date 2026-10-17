from fc import *
import json

def f14_debito():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"O modelo é só uma pequena parte de um sistema de ML real",ha="center",fontsize=11,fontweight="bold")
    B=[("Coleta de\ndados",2,60,22,26,GRY),("Verificação\nde dados",26,60,22,26,GRY),("Extração de\natributos",50,60,22,26,GRY),("Configuração",74,60,24,26,GRY),("Infraestrutura\ne recursos",2,8,30,26,GRY),("Monitoramento",66,8,32,26,GRY)]
    for t,x,y,w,h,c in B: rbox(ax,x,y,w,h,t,fc=MID,ec=GRY,fs=8,tc=INK,r=2)
    rbox(ax,36,20,26,36,"MODELO\n(código de ML)",fc=IND,ec=IND,fs=9.5,tc="white",bold=True,r=2)
    ax.text(50,2,"Adaptado da ideia de Sculley et al. (2015): a dívida técnica mora no entorno.",ha="center",fontsize=8,color=GRY,style="italic")
    save(fig,"f14_debito")

def f14_ciclo():
    fig,ax=canvas(8.6,4.2)
    ax.text(50,97,"O ciclo de vida de um modelo em produção",ha="center",fontsize=11,fontweight="bold")
    E=[("1 Dados",IND),("2 Treino\ne busca",CYA),("3 Validação",GRN),("4 Implantação",AMB),("5 Monitoramento",VIO),("6 Retreino",PNK)]
    cx,cy,R=50,50,33
    for i,(t,c) in enumerate(E):
        a=np.pi/2-i*2*np.pi/6; x=cx+R*1.35*np.cos(a); y=cy+R*0.85*np.sin(a)
        rbox(ax,x-10,y-7,20,14,t,fc=c,ec=c,fs=8.2,tc="white",bold=True,r=2)
    for i in range(6):
        a1=np.pi/2-i*2*np.pi/6; a2=np.pi/2-(i+1)*2*np.pi/6
        p1=(cx+R*1.35*np.cos(a1),cy+R*0.85*np.sin(a1)); p2=(cx+R*1.35*np.cos(a2),cy+R*0.85*np.sin(a2))
        mx=(p1[0]*0.62+p2[0]*0.38); my=(p1[1]*0.62+p2[1]*0.38); nx=(p1[0]*0.38+p2[0]*0.62); ny=(p1[1]*0.38+p2[1]*0.62)
        arrow(ax,(mx,my),(nx,ny),c=GRY,lw=1.8,ms=10)
    ax.text(50,50,"o ciclo\nnunca\ntermina",ha="center",va="center",fontsize=9.5,color=GRY,style="italic")
    save(fig,"f14_ciclo")

def f14_servir():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"Dois modos de usar um modelo",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,12,47,74,"",fc="white",ec=IND,lw=2.2,r=2); ax.text(25.5,78,"Em lote (batch)",ha="center",fontsize=10,fontweight="bold",color=IND)
    for i in range(5): rbox(ax,6+i*3.2,56-i*1.2,5,10,"",fc=MID,ec=IND,lw=1,r=1)
    arrow(ax,(25,50),(25,40),lw=2,ms=10); rbox(ax,12,28,27,10,"modelo",fc=IND,ec=IND,fs=8.6,tc="white",bold=True,r=1.5)
    ax.text(25.5,18,"muitos casos de uma vez, ex.: toda noite\nimporta a vazão, não a latência",ha="center",fontsize=7.8,color=INK)
    rbox(ax,51,12,47,74,"",fc="white",ec=GRN,lw=2.2,r=2); ax.text(74.5,78,"Em tempo real (online)",ha="center",fontsize=10,fontweight="bold",color=GRN)
    rbox(ax,58,52,12,10,"pedido",fc=MID,ec=GRN,fs=7.6,tc=INK,r=1); rbox(ax,76,52,16,10,"resposta",fc=GRN,ec=GRN,fs=7.6,tc="white",bold=True,r=1)
    rbox(ax,64,30,20,10,"modelo",fc=GRN,ec=GRN,fs=8.6,tc="white",bold=True,r=1.5); arrow(ax,(64,51),(70,41),lw=1.6,ms=9); arrow(ax,(80,41),(84,51),lw=1.6,ms=9)
    ax.text(74.5,18,"um caso por vez, resposta em ms\na latência é crítica",ha="center",fontsize=7.8,color=INK)
    save(fig,"f14_servir")

def f14_deriva():
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.0)); rng=np.random.default_rng(1)
    a=axs[0]; xs=np.linspace(-3,5,200)
    f=lambda m,s:np.exp(-(xs-m)**2/(2*s*s))/(s*np.sqrt(2*np.pi))
    a.fill_between(xs,f(0,1),color=IND,alpha=.35,label="treino"); a.fill_between(xs,f(1.8,1),color=AMB,alpha=.35,label="produção")
    a.set_title("Deriva dos dados: os atributos mudam",fontsize=9.3); a.set_yticks([]); a.set_xlabel("valor do atributo (ex.: renda)"); a.legend(frameon=False,fontsize=8)
    b=axs[1]; x=rng.uniform(0,1,60)
    b.scatter(x,0.2+0.5*x+rng.normal(0,.06,60),s=14,color=IND,label="antes"); b.scatter(x,0.7-0.5*x+rng.normal(0,.06,60),s=14,color=AMB,label="depois")
    b.set_title("Deriva de conceito: a relação muda",fontsize=9.3); b.set_xlabel("atributo"); b.set_ylabel("resposta"); b.legend(frameon=False,fontsize=8)
    fig.tight_layout(); save(fig,"f14_deriva")

def f14_monitor():
    d=json.load(open("/home/user/claude/automl/build/f14_deriva.json")); m=[r["mes"] for r in d]
    fig,ax=plt.subplots(figsize=(6.8,3.5))
    ax.plot(m,[r["psi_renda"] for r in d],"-o",color=RED,lw=2.2,label="PSI da renda (muda)"); ax.plot(m,[r["psi_emp"] for r in d],"-o",color=GRY,lw=2,label="PSI do tempo de emprego (estável)")
    ax.axhline(0.10,color=AMB,ls="--",lw=1); ax.text(0.05,0.115,"alerta (0,10)",fontsize=7.5,color=AMB); ax.axhline(0.25,color=RED,ls="--",lw=1); ax.text(0.05,0.265,"mudança grande (0,25)",fontsize=7.5,color=RED)
    ax.set_xlabel("meses após a implantação"); ax.set_ylabel("PSI")
    a2=ax.twinx(); a2.plot(m,[r["acc"] for r in d],"-s",color=IND,lw=2,label="acurácia (eixo direito)"); a2.set_ylabel("acurácia contra a verdade",color=IND); a2.set_ylim(0.78,0.88); a2.spines["right"].set_visible(True)
    h1,l1=ax.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,frameon=False,fontsize=7.5,loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=2)
    ax.set_title("Alarme nos dados antes da queda de desempenho",fontsize=10); save(fig,"f14_monitor")

def f14_campeao():
    fig,ax=canvas(8.6,3.7)
    ax.text(50,97,"Campeão e desafiante",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,50,26,30,"Campeão\n(em produção)",fc=IND,ec=IND,fs=9,tc="white",bold=True,r=2); rbox(ax,2,12,26,30,"Desafiante\n(retreinado com\ndados recentes)",fc=AMB,ec=AMB,fs=8.4,tc="white",bold=True,r=2)
    rbox(ax,38,28,26,34,"Desafio nos\ndados recentes\ncom rótulos",fc="white",ec=GRY,fs=8.6,tc=INK,lw=2,r=2)
    arrow(ax,(28.5,65),(37.5,50),lw=2,ms=11); arrow(ax,(28.5,27),(37.5,40),lw=2,ms=11)
    rbox(ax,74,52,24,28,"ganha com margem?\nSIM → desafiante\nassume",fc="#ECFDF5",ec=GRN,fs=8,tc=INK,lw=2,r=2); rbox(ax,74,14,24,28,"NÃO → mantém\no campeão",fc="#FEF2F2",ec=RED,fs=8.2,tc=INK,lw=2,r=2)
    arrow(ax,(64.5,50),(73.5,62),lw=2,ms=11,c=GRN); arrow(ax,(64.5,40),(73.5,30),lw=2,ms=11,c=RED)
    save(fig,"f14_campeao")

def f14_roteiro():
    fig,ax=canvas(8.6,4.2)
    ax.text(50,97,"Roteiro completo de um projeto de AutoML",ha="center",fontsize=11,fontweight="bold")
    S=["1 Problema\ne métrica","2 Teste final\n+ vazamento","3 Linhas\nde base","4 Pipeline","5 Busca","6 Ensemble +\nferramentas","7 Teste único,\npor grupo","8 Explicar\ne auditar","9 Implantar\n+ ficha","10 Monitorar\n+ retreinar"]
    C=[IND,IND,CYA,CYA,GRN,GRN,AMB,AMB,VIO,VIO]
    for i,(t,c) in enumerate(zip(S,C)):
        x=2+(i%5)*19.6; y=52 if i<5 else 10
        rbox(ax,x,y,17.6,34,t,fc=c,ec=c,fs=8.2,tc="white",bold=True,r=2)
        if i%5<4: arrow(ax,(x+17.9,y+17),(x+19.5,y+17),lw=1.6,ms=9)
    arrow(ax,(90,51),(11,45.5),c=GRY,lw=1.2,ms=1) if False else None
    save(fig,"f14_roteiro")

def f14_jornada():
    fig,ax=canvas(8.6,4.6)
    ax.text(50,97,"A jornada deste livro",ha="center",fontsize=11,fontweight="bold")
    P=[("Parte I","Fundamentos\nCaps. 1–3",IND),("Parte II","Busca de\nhiperparâmetros\nCaps. 4–7",CYA),("Parte III","Automação do\npipeline\nCaps. 8–10",GRN),("Parte IV","Redes e novos\ndomínios\nCaps. 11–12",AMB),("Parte V","Responsabilidade\ne prática\nCaps. 13–14",VIO)]
    for i,(t,d,c) in enumerate(P):
        x=2+i*19.6; h=24+i*11
        rbox(ax,x,8,17.6,h+10,"",fc=c,ec=c,r=2); ax.text(x+8.8,8+h+5,t,ha="center",fontsize=9,fontweight="bold",color="white"); ax.text(x+8.8,8+(h+10)/2-3,d,ha="center",va="center",fontsize=7.8,color="white",linespacing=1.3)
    ax.text(50,3,"Cada degrau reaproveita o anterior: o AutoML é uma orquestra de ideias.",ha="center",fontsize=8.6,color=GRY,style="italic")
    save(fig,"f14_jornada")

if __name__=="__main__":
    for fn_ in (f14_debito,f14_ciclo,f14_servir,f14_deriva,f14_monitor,f14_campeao,f14_roteiro,f14_jornada): fn_()

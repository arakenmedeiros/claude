from icons import *
import sys

def f1_central_vs_fed():
    fig,axs=plt.subplots(1,2,figsize=(10,4.6)); ASP[0]=(5/4.6)
    for ax in axs: ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
    pos=[(14,80,"phone"),(14,55,"hospital"),(14,30,"bank"),(14,8,"phone")]
    def draw(ax,x,y,k,c):
        {"phone":phone,"hospital":hospital,"bank":bank}[k](ax,x,y,0.9,c)
    cols=[IND,RED,AMB,PNK]
    # esquerda: centralizado
    ax=axs[0]; ax.set_title("Aprendizado centralizado",loc="left",fontsize=12,color=INK)
    cloud(ax,78,52,1.9,CYA); server(ax,78,52,1.1); ax.text(78,26,"Servidor com\nTODOS os dados",ha="center",fontsize=9.5,fontweight="bold")
    for (x,y,k),c in zip(pos,cols):
        draw(ax,x,y,k,c); db(ax,36,y,0.6,c)
        arrow(ax,(41,y),(60,52+(y-52)*0.35),c=RED,lw=2)
    ax.text(48,96,"os dados brutos viajam",ha="center",color=RED,fontsize=9.5,fontweight="bold")
    # direita: federado
    ax=axs[1]; ax.set_title("Aprendizado federado",loc="left",fontsize=12,color=INK)
    cloud(ax,78,52,1.9,GRN); server(ax,78,52,1.1); ax.text(78,26,"Servidor só vê\nmodelos",ha="center",fontsize=9.5,fontweight="bold")
    for (x,y,k),c in zip(pos,cols):
        draw(ax,x,y,k,c); db(ax,30,y,0.5,c); lock(ax,30,y-6.5,0.6,GRN); netglyph(ax,43,y,0.55,VIO)
        arrow(ax,(50,y+2.5),(60,52+(y-52)*0.35+2),c=GRN,lw=2); arrow(ax,(60,52+(y-52)*0.35-2),(50,y-2.5),c=VIO,lw=1.4,ls="--")
    ax.text(48,96,"só os modelos viajam",ha="center",color=GRN,fontsize=9.5,fontweight="bold")
    save(fig,"f1_central_vs_fed")

def f1_exemplos():
    fig,ax=canvas(10,3.8)
    cards=[("Teclado do celular",IND,phone,"Sugestão da próxima palavra","Suas mensagens ficam\nno aparelho"),
           ("Hospitais",RED,hospital,"Detecção de doenças em exames","Prontuários e imagens\nficam no hospital"),
           ("Bancos",AMB,bank,"Detecção de fraude","Transações dos clientes\nficam no banco")]
    for i,(t,c,ic,tarefa,fica) in enumerate(cards):
        x=3+i*32.5
        rbox(ax,x,6,30,88,fc="white",ec=c,lw=2.2,r=3)
        rbox(ax,x,78,30,16,t,fc=c,tc="white",bold=True,fs=11,r=3)
        ic(ax,x+15,60,1.5,c)
        ax.text(x+15,44,"Tarefa",ha="center",fontsize=8,color=GRY)
        ax.text(x+15,37,tarefa.replace(" de ","\nde ",1) if len(tarefa)>22 else tarefa,ha="center",va="center",fontsize=8.5,fontweight="bold",linespacing=1.2)
        ax.text(x+15,25,"O que fica local",ha="center",fontsize=8,color=GRY)
        ax.text(x+15,15,fica,ha="center",va="center",fontsize=8.5,color=INK,linespacing=1.2)
    save(fig,"f1_exemplos")

def f1_historia():
    fig,ax=canvas(10,3.6,(0,100),(0,100))
    ax.plot([4,96],[50,50],c=INK,lw=3)
    ev=[(8,"2012","Big data e\nredes profundas\nganham força",1,GRY),
        (24,"2016","Termo 'aprendizado\nfederado' nasce\nno Google",-1,IND),
        (38,"2017","Artigo FedAvg e\nagregação segura\n(Bonawitz)",1,CYA),
        (52,"2018","GDPR (UE) e\nLei 13.709 (LGPD)\nno Brasil",-1,RED),
        (66,"2019","Yang et al. (ACM TIST)\ne sistema em escala\n(Bonawitz et al.)",1,GRN),
        (80,"2021","Kairouz et al.:\n'Advances and\nOpen Problems'",-1,VIO),
        (92,"2023+","Modelos de\nlinguagem\nfederados",1,PNK)]
    for x,a,t,s,c in ev:
        ax.add_patch(Circle((x,50),2.4,fc=c,ec="white",lw=2,zorder=3))
        ax.plot([x,x],[50,50+s*14],c=c,lw=2)
        ax.text(x,50+s*18,a,ha="center",va="center",fontsize=11,fontweight="bold",color=c)
        ax.text(x,50+s*(31 if s>0 else 31)*1.0-(0 if s>0 else 0),t,ha="center",va="bottom" if s>0 else "top",fontsize=7.8,color=INK,linespacing=1.2) if False else None
        ax.text(x,50+s*24,t,ha="center",va="bottom" if s>0 else "top",fontsize=7.8,color=INK,linespacing=1.2)
    save(fig,"f1_historia")

def f1_cross():
    fig,ax=canvas(10,4.4)
    rbox(ax,2,52,46,44,fc="#EEF2FF",ec=IND,lw=2,r=3); rbox(ax,52,52,46,44,fc="#FEF2F2",ec=RED,lw=2,r=3)
    ax.text(25,90,"Cross-device (entre dispositivos)",ha="center",fontsize=11,fontweight="bold",color=IND)
    ax.text(75,90,"Cross-silo (entre organizações)",ha="center",fontsize=11,fontweight="bold",color=RED)
    rng=np.random.default_rng(1)
    for i in range(34):
        phone(ax,rng.uniform(6,44),rng.uniform(56,84),0.33,IND)
    for i,(x,y) in enumerate([(62,68),(75,76),(88,68),(68,58),(84,58)]):
        hospital(ax,x,y,0.45,RED)
    rows=[("Clientes","milhões de celulares, relógios, carros","poucas dezenas de hospitais, bancos"),
          ("Disponibilidade","instável: só carregando e em Wi-Fi","alta: servidores sempre ligados"),
          ("Dados por cliente","poucos","muitos"),
          ("Principal desafio","comunicação e dispositivos que somem","governança, confiança e dados diferentes")]
    for i,(a,b,c) in enumerate(rows):
        y=44-i*11
        rbox(ax,2,y-4,18,9,a,fc=INK,tc="white",fs=8.5,bold=True,r=1.5)
        rbox(ax,21,y-4,38,9,b,fc="#EEF2FF",fs=8,r=1.5); rbox(ax,60,y-4,38,9,c,fc="#FEF2F2",fs=8,r=1.5)
    save(fig,"f1_cross")

def f1_roteiro():
    fig,ax=canvas(10,3.4)
    partes=[("I","Fundamentos","Caps. 1–3",IND),("II","Desafios do\ntreino","Caps. 4–6",CYA),("III","Privacidade e\nsegurança","Caps. 7–9",RED),("IV","Arquiteturas e\nfronteiras","Caps. 10–11",AMB),("V","Projeto final","Cap. 12",VIO)]
    for i,(n,t,c,col) in enumerate(partes):
        x=2+i*19.6
        rbox(ax,x,22,18,60,fc=col,r=3,alpha=0.12,ec=col,lw=2)
        ax.add_patch(Circle((x+9,70),3.4,fc=col,ec="white",lw=2)); ax.text(x+9,70,n,ha="center",va="center",fontsize=14,fontweight="bold",color="white")
        ax.text(x+9,47,t,ha="center",va="center",fontsize=10,fontweight="bold",color=INK,linespacing=1.2)
        ax.text(x+9,29,c,ha="center",va="center",fontsize=9,color=GRY)
        if i<4: arrow(ax,(x+18.2,52),(x+19.4,52),c=INK,lw=1.5,ms=8)
    ax.text(50,10,"do básico ao avançado, sempre com prática em Python",ha="center",fontsize=10.5,style="italic",color=GRY)
    save(fig,"f1_roteiro")

def f1_ciclo():
    fig,ax=canvas(8,4.2)
    server(ax,50,50,1.0)
    ax.text(50,34,"Servidor",ha="center",fontsize=10,fontweight="bold")
    cl=[(14,80,IND,phone),(86,80,RED,hospital),(14,18,AMB,bank),(86,18,PNK,phone)]
    for x,y,c,ic in cl:
        ic(ax,x,y,0.9,c)
    steps=[((26,72),(40,58),"①"),((60,58),(74,72),"③"),((26,26),(40,42),"②"),((60,42),(74,26),"④")]
    for (a,b),n in zip([((40,58),(24,74)),((76,74),(60,58)),((24,24),(40,42)),((60,42),(76,26))],"1342"): pass
    arrow(ax,(40,57),(25,72),c=VIO,lw=2); arrow(ax,(40,43),(25,27),c=VIO,lw=2)
    arrow(ax,(75,72),(60,57),c=GRN,lw=2); arrow(ax,(75,27),(60,43),c=GRN,lw=2)
    ax.text(27,56,"1 envia o\nmodelo",fontsize=8.5,color=VIO,ha="center",fontweight="bold")
    ax.text(73,56,"3 devolvem\natualizações",fontsize=8.5,color=GRN,ha="center",fontweight="bold")
    ax.text(50,92,"2 cada cliente treina com seus dados locais",ha="center",fontsize=9,color=INK,fontweight="bold")
    ax.text(50,6,"4 o servidor combina tudo e começa nova rodada",ha="center",fontsize=9,color=INK,fontweight="bold")
    save(fig,"f1_ciclo")

if __name__=="__main__":
    for f in [f1_central_vs_fed,f1_exemplos,f1_historia,f1_cross,f1_roteiro,f1_ciclo]: f()

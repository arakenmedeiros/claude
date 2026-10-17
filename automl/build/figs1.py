from fc import *

def f1_cozinha():
    fig,ax=canvas(8.6,4.3)
    ax.text(50,96,"A cozinha do aprendizado de máquina: muitas receitas possíveis",ha="center",fontsize=11,fontweight="bold")
    # left: human chef
    rbox(ax,2,50,40,40,fc="#EEF2FF",ec=IND,lw=1.5)
    ax.text(22,86,"Cozinheiro humano",ha="center",fontsize=10,fontweight="bold",color=IND)
    ax.text(22,79,"experiência + intuição",ha="center",fontsize=8.5,color=GRY)
    for i,(t,c) in enumerate([("testa 5 receitas",IND),("ajusta o sal à mão",IND),("leva dias",RED)]):
        rbox(ax,6,68-i*8.2,32,6.2,t,fc="white",ec=c,fs=8.5,tc=INK)
    # right: automated
    rbox(ax,58,50,40,40,fc="#ECFEFF",ec=CYA,lw=1.5)
    ax.text(78,86,"Cozinheiro automático",ha="center",fontsize=10,fontweight="bold",color="#0E7490")
    ax.text(78,79,"busca sistemática",ha="center",fontsize=8.5,color=GRY)
    for i,(t,c) in enumerate([("testa 500 receitas",CYA),("ajusta tudo com método",CYA),("leva horas",GRN)]):
        rbox(ax,62,68-i*8.2,32,6.2,t,fc="white",ec=c,fs=8.5,tc=INK)
    arrow(ax,(43,70),(57,70),c=GRY,lw=2); ax.text(50,73,"automatiza",ha="center",fontsize=8,color=GRY,style="italic")
    # bottom: recipe space grid
    ax.text(50,43,"Cada combinação de ingredientes e temperos é um ponto no espaço de receitas",ha="center",fontsize=9,color=INK)
    rng=np.random.default_rng(3)
    gx,gy=np.meshgrid(np.linspace(8,92,14),np.linspace(8,34,6)); gx=gx.ravel(); gy=gy.ravel()
    z=np.exp(-(((gx-68)/18)**2+((gy-24)/9)**2))+0.5*np.exp(-(((gx-25)/14)**2+((gy-14)/8)**2))
    ax.scatter(gx,gy,c=z,cmap="YlGnBu",s=70,edgecolors=MID,linewidths=0.6)
    b=z.argmax(); ax.scatter([gx[b]],[gy[b]],s=380,marker="*",c=AMB,edgecolors="white",linewidths=1.2,zorder=5)
    ax.annotate("melhor receita",(gx[b],gy[b]),(84,2),fontsize=8.5,color=INK,ha="center",arrowprops=dict(arrowstyle="-",color=GRY))
    save(fig,"f1_cozinha")

def f1_pipeline():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,96,"As etapas de um projeto de aprendizado de máquina e o que o AutoML automatiza",ha="center",fontsize=10.5,fontweight="bold")
    steps=[("Coletar\ne limpar\ndados",0,GRY),("Preparar\n(imputar,\nescalar)",1,IND),("Criar e\nselecionar\natributos",2,CYA),("Escolher\no algoritmo",3,GRN),("Ajustar\nhiperpar.",4,AMB),("Combinar\nmodelos",5,PNK),("Avaliar e\nimplantar",6,VIO)]
    w=12.2; gap=2.1
    for t,i,c in steps:
        x=2+i*(w+gap)
        rbox(ax,x,48,w,30,t,fc=c,ec=c,fs=8.2,tc="white",bold=True,r=2)
        if i<6: arrow(ax,(x+w+0.1,63),(x+w+gap-0.1,63),c=INK,lw=1.3,ms=9)
    # brackets
    def brace(x0,x1,y,label,c):
        ax.plot([x0,x0,x1,x1],[y+3,y,y,y+3],c=c,lw=2); ax.text((x0+x1)/2,y-6,label,ha="center",fontsize=8.5,color=c,fontweight="bold")
    brace(2+1*(w+gap),2+2*(w+gap)+w,36,"AutoML de dados e atributos",CYA)
    brace(2+3*(w+gap),2+5*(w+gap)+w,24,"AutoML clássico: CASH + ensembles",GRN)
    brace(2+3*(w+gap),2+4*(w+gap)+w,12,"HPO",AMB)
    ax.text(2+0*(w+gap)+w/2,36,"quase sempre\nhumano",ha="center",fontsize=8,color=GRY,va="center",style="italic")
    ax.text(2+6*(w+gap)+w/2,36,"MLOps\n(Cap. 14)",ha="center",fontsize=8,color=VIO,va="center",style="italic")
    ax.text(50,84,"Linha de montagem do modelo",ha="center",fontsize=8.5,color=GRY,style="italic")
    save(fig,"f1_pipeline")

def f1_cash():
    fig,ax=canvas(8.6,4.6)
    ax.text(50,97,"O problema CASH: escolher o algoritmo e, ao mesmo tempo, seus ajustes",ha="center",fontsize=10.5,fontweight="bold")
    rbox(ax,38,82,24,8,"Algoritmo?",fc=INK,fs=10,tc="white",bold=True)
    algs=[("Árvore de\ndecisão",8,IND,["profundidade máx.","amostras por folha"]),("k-vizinhos",30,CYA,["nº de vizinhos","tipo de peso"]),("Regressão\nlogística",52,GRN,["força C","penalidade"]),("Floresta\naleatória",74,AMB,["nº de árvores","atributos por divisão"])]
    for t,x,c,hp in algs:
        rbox(ax,x,56,18,12,t,fc=c,ec=c,fs=8.6,tc="white",bold=True)
        arrow(ax,(50,81.5),(x+9,69),c=GRY,lw=1.2,ms=9)
        for j,h in enumerate(hp):
            rbox(ax,x-1.5,40-j*10.5,21,8,h,fc="white",ec=c,fs=7.6,tc=INK,lw=1.2)
            ax.plot([x+9,x+9],[56,48.2] if j==0 else [40,40-0.0],c=c,lw=1.2)
        ax.plot([x+9,x+9],[56,48],c=c,lw=1.2)
    ax.text(50,16,"Hiperparâmetros (os \"botões\" do algoritmo) dependem do algoritmo escolhido.",ha="center",fontsize=9,color=INK)
    ax.text(50,9,"É um espaço de busca em árvore: escolher um ramo muda quais botões existem.",ha="center",fontsize=9,color=GRY,style="italic")
    save(fig,"f1_cash")

def f1_historia():
    fig,ax=plt.subplots(figsize=(8.6,4.6)); ax.set_xlim(2011.6,2026.4); ax.set_ylim(-4.6,4.4); ax.axis("off")
    ax.plot([2012,2026],[0,0],c=INK,lw=3)
    ev=[(2013,"Auto-WEKA","primeiro CASH formal",IND,1.0),(2015,"Auto-sklearn","meta-aprendizado\n+ ensembles",CYA,-1.0),(2016,"TPOT","pipelines por\nprog. genética",GRN,2.4),(2017,"NAS com RL","redes projetadas\npor máquinas",AMB,-2.6),(2018,"Hyperband\nBOHB","busca\nmultifidelidade",RED,1.0),(2019,"Optuna\nDARTS","HPO flexível;\nNAS diferenciável",VIO,-1.0),(2020,"AutoGluon\nH2O AutoML","ensembles\nem camadas",PNK,2.4),(2021,"FLAML","AutoML\neconômico",IND,-2.6),(2022,"Auto-sklearn 2.0","sem\nmeta-atributos",CYA,1.0),(2024,"AMLB","benchmark\naberto",GRN,-1.0),(2025,"TabPFN","modelo-fundação\ntabular",AMB,2.4)]
    for y,t,d,c,s_ in ev:
        ax.plot([y,y],[0,s_],c=c,lw=1.4); ax.scatter([y],[0],s=120,c=c,edgecolors="white",linewidths=1.5,zorder=3)
        up=s_>0; nl=t.count("\n")+1
        ax.text(y,s_+(0.12 if up else -0.12),t,ha="center",va="bottom" if up else "top",fontsize=8,fontweight="bold",color=c,linespacing=1.1)
        off=0.12+0.42*nl+0.1
        ax.text(y,s_+(off if up else -off),d,ha="center",va="bottom" if up else "top",fontsize=7,color=INK,linespacing=1.15)
    ax.text(2011.7,4.4,"Uma linha do tempo do AutoML",fontsize=11,fontweight="bold",va="top")
    for y in (2013,2015,2017,2019,2021,2023,2025): ax.text(y,-0.28,str(y),ha="center",va="top",fontsize=6.5,color=GRY) if False else None
    save(fig,"f1_historia")

def f1_niveis():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"Níveis de automação do aprendizado de máquina (como no carro autônomo)",ha="center",fontsize=10,fontweight="bold")
    ax.text(50,90,"As ferramentas de hoje estão, em geral, entre os níveis 2 e 3",ha="center",fontsize=9,color=GRY,style="italic")
    L=[("0","Tudo manual","você escolhe\ne ajusta tudo",GRY),("1","Ajuda pontual","busca em grade\nou aleatória",IND),("2","Busca inteligente","otimização bayesiana,\nHyperband",CYA),("3","Pipeline completo","algoritmo + atributos\n+ ensemble",GRN),("4","Ciclo completo","do dado à\nimplantação",AMB),("5","Autonomia total","ainda não existe\nna prática",RED)]
    for i,(n,t,d,c) in enumerate(L):
        x=3+i*15.7; h=10+i*7.5
        rbox(ax,x,28,14,h,fc=c,ec=c,r=1.5,alpha=0.92)
        ax.text(x+7,28+h-4.5,n,ha="center",fontsize=17,fontweight="bold",color="white")
        ax.text(x+7,23,t,ha="center",va="center",fontsize=7.1,fontweight="bold",color=c)
        ax.text(x+7,12.5,d,ha="center",va="center",fontsize=6.9,color=INK,linespacing=1.15)
    save(fig,"f1_niveis")

def f1_papeis():
    fig,ax=canvas(8.6,3.7)
    ax.text(50,96,"Quem faz o quê? Pessoa e AutoML trabalham juntos",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,3,20,42,66,fc="#EEF2FF",ec=IND,lw=1.5); rbox(ax,55,20,42,66,fc="#ECFEFF",ec=CYA,lw=1.5)
    ax.text(24,81,"A pessoa",ha="center",fontsize=11,fontweight="bold",color=IND); ax.text(76,81,"O AutoML",ha="center",fontsize=11,fontweight="bold",color="#0E7490")
    A=["Formula o problema","Entende o contexto","Define a métrica certa","Cuida de ética e vieses","Decide se confia no resultado"]
    B=["Testa muitas combinações","Ajusta hiperparâmetros","Monta ensembles","Não se cansa nem se esquece","Registra cada experimento"]
    for i,(a,b) in enumerate(zip(A,B)):
        rbox(ax,6,66-i*10.5,36,8,a,fc="white",ec=IND,fs=8.5); rbox(ax,58,66-i*10.5,36,8,b,fc="white",ec=CYA,fs=8.5)
    ax.annotate("",xy=(54,52),xytext=(46,52),arrowprops=dict(arrowstyle="<->",color=AMB,lw=2.5))
    ax.text(50,10,"AutoML não substitui o julgamento humano: ele amplia o alcance dele.",ha="center",fontsize=9,color=INK,style="italic")
    save(fig,"f1_papeis")

if __name__=="__main__":
    for f in (f1_cozinha,f1_pipeline,f1_cash,f1_historia,f1_niveis,f1_papeis): f()

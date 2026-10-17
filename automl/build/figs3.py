from fc import *
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier

def f3_lavanderia():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"A lavanderia dos dados: da roupa suja ao modelo",ha="center",fontsize=11,fontweight="bold")
    items=[("Valores\nfaltando",RED,"imputar"),("Escalas\ndiferentes",AMB,"padronizar"),("Categorias\nem texto",CYA,"codificar"),("Atributos\ninúteis",VIO,"selecionar")]
    for i,(t,c,a) in enumerate(items):
        x=3+i*16
        rbox(ax,x,56,14,20,t,fc="white",ec=c,fs=8.6,tc=INK,lw=2); ax.text(x+7,50.5,a,ha="center",fontsize=8.3,color=c,fontweight="bold")
        arrow(ax,(x+7,47.2),(x+7,43.8),c=c,lw=1.4,ms=8)
    rbox(ax,3,30,62,13,"Pipeline de pré-processamento",fc=IND,ec=IND,fs=10,tc="white",bold=True,r=2)
    arrow(ax,(66,36.5),(76,36.5),c=INK,lw=2,ms=12)
    rbox(ax,77,26,20,22,"Modelo\nlimpo e\npronto",fc=GRN,ec=GRN,fs=9.5,tc="white",bold=True,r=2)
    ax.text(50,12,"Cada tipo de problema tem sua ferramenta. O pipeline aplica todas, sempre na mesma ordem.",ha="center",fontsize=9,color=INK)
    ax.text(50,5,"Dados sujos geram modelos ruins, não importa o quão sofisticado seja o AutoML.",ha="center",fontsize=8.5,color=GRY,style="italic")
    save(fig,"f3_lavanderia")

def f3_onehot():
    fig,ax=canvas(8.6,3.3)
    ax.text(50,96,"Codificação one-hot: transformando texto em números sem inventar ordem",ha="center",fontsize=10.5,fontweight="bold")
    ax.text(41,60,"novas colunas: uma por cidade",ha="center",fontsize=8.3,color="#B45309",fontweight="bold"); rbox(ax,4,48,10,8,"cidade",fc=IND,ec="white",fs=8.6,tc="white",bold=True,r=0.6)
    for i,v in enumerate(["Recife","Natal","Recife","Salvador"]): rbox(ax,4,38-i*9,10,8,v,fc="#EEF2FF",ec="white",fs=8.3,r=0.6)
    arrow(ax,(16,32),(26,32),c=INK,lw=2,ms=12)
    for j,c in enumerate(["Recife","Natal","Salvador"]): rbox(ax,28+j*14,48,13,8,c,fc=AMB,ec="white",fs=8.3,tc="white",bold=True,r=0.6)
    M=[[1,0,0],[0,1,0],[1,0,0],[0,0,1]]
    for i,r in enumerate(M):
        for j,v in enumerate(r): rbox(ax,28+j*14,38-i*9,13,8,str(v),fc="#FEF3C7" if v else "#F9FAFB",ec="white",fs=9,fw=None) if False else rbox(ax,28+j*14,38-i*9,13,8,str(v),fc="#FDE68A" if v else "#F9FAFB",ec="white",fs=9,r=0.6,bold=bool(v))
    ax.text(79,36,"Por que não usar\n1, 2, 3?\nPorque o modelo\nentenderia que\nSalvador > Natal\n> Recife.",fontsize=8.2,color=INK,va="center",linespacing=1.25)
    ax.text(50,6,"Cada categoria vira uma coluna de 0 ou 1 (\"está presente?\").",ha="center",fontsize=9,color=GRY,style="italic")
    save(fig,"f3_onehot")

def f3_colunas():
    fig,ax=canvas(8.6,4.4)
    ax.text(50,97,"Pipeline com ColumnTransformer: cada tipo de coluna tem seu tratamento",ha="center",fontsize=10.5,fontweight="bold")
    rbox(ax,2,42,13,16,"Tabela\nbruta",fc=GRY,ec=GRY,fs=9,tc="white",bold=True,r=2)
    # branches
    rbox(ax,22,68,16,12,"Imputar\n(mediana)",fc=IND,ec=IND,fs=8.5,tc="white",bold=True); rbox(ax,42,68,16,12,"Padronizar\n(escala)",fc=IND,ec=IND,fs=8.5,tc="white",bold=True)
    rbox(ax,22,22,16,12,"Imputar\n(mais comum)",fc=CYA,ec=CYA,fs=8.5,tc="white",bold=True); rbox(ax,42,22,16,12,"One-hot",fc=CYA,ec=CYA,fs=8.5,tc="white",bold=True)
    ax.text(40,84,"colunas numéricas",ha="center",fontsize=9,color=IND,fontweight="bold"); ax.text(40,17,"colunas categóricas",ha="center",fontsize=9,color="#0E7490",fontweight="bold")
    arrow(ax,(15.5,53),(21.5,72),c=INK,lw=1.4,ms=9); arrow(ax,(15.5,47),(21.5,28),c=INK,lw=1.4,ms=9)
    arrow(ax,(38.3,74),(41.7,74),c=INK,lw=1.4,ms=8); arrow(ax,(38.3,28),(41.7,28),c=INK,lw=1.4,ms=8)
    rbox(ax,66,42,13,16,"Juntar\ncolunas",fc=AMB,ec=AMB,fs=9,tc="white",bold=True,r=2)
    arrow(ax,(58.3,74),(66,56),c=INK,lw=1.4,ms=9); arrow(ax,(58.3,28),(66,44),c=INK,lw=1.4,ms=9)
    arrow(ax,(79.3,50),(84,50),c=INK,lw=1.6,ms=10)
    rbox(ax,84.5,42,13.5,16,"Modelo",fc=GRN,ec=GRN,fs=10,tc="white",bold=True,r=2)
    ax.plot([1,99],[10,10],c=MID,lw=1); ax.text(50,5,"Tudo isso é um único objeto Pipeline: um só .fit() e um só .predict().",ha="center",fontsize=9,color=INK,style="italic")
    save(fig,"f3_colunas")

def f3_vazamento():
    fig,ax=canvas(8.6,4.1)
    ax.text(50,97,"Vazamento de dados: o erro que faz a nota parecer melhor do que é",ha="center",fontsize=11,fontweight="bold")
    # wrong
    ax.text(3,86,"ERRADO",fontsize=10,fontweight="bold",color=RED)
    rbox(ax,3,72,26,10,"1. Calcular a escala\ncom TODOS os dados",fc="#FEE2E2",ec=RED,fs=8.3); arrow(ax,(30,77),(35,77),ms=9)
    rbox(ax,36,72,26,10,"2. Dividir em\ntreino e validação",fc="#FEE2E2",ec=RED,fs=8.3); arrow(ax,(63,77),(68,77),ms=9)
    rbox(ax,69,72,28,10,"3. A validação já\n\"espiou\" o treino",fc=RED,ec=RED,fs=8.3,tc="white",bold=True)
    # right
    ax.text(3,56,"CERTO",fontsize=10,fontweight="bold",color=GRN)
    rbox(ax,3,42,26,10,"1. Dividir em treino\ne validação",fc="#D1FAE5",ec=GRN,fs=8.3); arrow(ax,(30,47),(35,47),ms=9)
    rbox(ax,36,42,26,10,"2. Calcular a escala\nSÓ com o treino",fc="#D1FAE5",ec=GRN,fs=8.3); arrow(ax,(63,47),(68,47),ms=9)
    rbox(ax,69,42,28,10,"3. Aplicar a mesma\nescala na validação",fc=GRN,ec=GRN,fs=8.3,tc="white",bold=True)
    ax.text(50,26,"Analogia: estudar com a prova na mão. A nota sobe, mas não mede aprendizado.",ha="center",fontsize=9,color=INK)
    ax.text(50,16,"Regra prática: toda etapa que \"aprende\" algo dos dados (escala, imputação, seleção)\ndeve estar dentro do Pipeline e ser reaprendida a cada divisão da validação cruzada.",ha="center",fontsize=8.6,color=GRY,style="italic",linespacing=1.3)
    save(fig,"f3_vazamento")

def f3_escalas():
    X,y=load_wine(return_X_y=True); a=X[:,[0,12]]  # alcohol, proline
    from sklearn.preprocessing import StandardScaler
    az=StandardScaler().fit_transform(a)
    fig,axs=plt.subplots(1,2,figsize=(8.6,3.3))
    for ax,(D,t,xl,yl) in zip(axs,[(a,"Sem padronizar","álcool (11 a 15)","prolina (278 a 1680)"),(az,"Padronizado","álcool (padronizado)","prolina (padronizada)")]):
        ax.scatter(D[:,0],D[:,1],c=[[IND,AMB,GRN][k] for k in y],s=14,edgecolors="white",linewidths=0.3)
        q=D[np.argmin(np.abs(D[:,0]-D[:,0].mean())+np.abs(D[:,1]-np.median(D[:,1])))]
        ax.scatter(*q,s=130,marker="*",c=RED,edgecolors="white",zorder=5)
        d=np.linalg.norm((D-q)/ (D.std(0) if D is a else 1),axis=1)
        ax.set_title(t,fontsize=10,fontweight="bold"); ax.set_xlabel(xl,fontsize=8); ax.set_ylabel(yl,fontsize=8)
    axs[0].text(0.03,0.93,"a prolina domina\nqualquer distância",transform=axs[0].transAxes,fontsize=8.5,color=RED,fontweight="bold",va="top")
    axs[1].text(0.03,0.93,"as duas medidas\npesam parecido",transform=axs[1].transAxes,fontsize=8.5,color=GRN,fontweight="bold",va="top")
    fig.suptitle("Por que padronizar: números grandes \"gritam\" mais alto que números pequenos",fontsize=10,fontweight="bold",y=1.02)
    save(fig,"f3_escalas")

def f3_espaco():
    fig,ax=canvas(8.6,4.5)
    ax.text(50,97,"O pipeline inteiro é o espaço de busca do AutoML",ha="center",fontsize=11,fontweight="bold")
    cols=[("Imputação",2,IND,["média","mediana","mais comum"]),("Escala",26,CYA,["padrão","min-máx","nenhuma"]),("Seleção",50,GRN,["nenhuma","top-k","PCA"]),("Algoritmo",74,AMB,["árvore","k-vizinhos","logística"])]
    for t,x,c,ops in cols:
        rbox(ax,x,78,22,9,t,fc=c,ec=c,fs=9.5,tc="white",bold=True)
        for k,o in enumerate(ops): rbox(ax,x+1,64-k*11,20,8.5,o,fc="white",ec=c,fs=8.3,lw=1.4)
        ax.plot([x+11,x+11],[78,72.5],c=c,lw=1.3)
    for i in range(3):
        x=2+i*24+22; ax.annotate("",xy=(x+2,82),xytext=(x,82),arrowprops=dict(arrowstyle="-|>",color=INK,lw=1.4))
    ax.text(85,25,"+ hiperparâmetros\nde cada um",ha="center",fontsize=8.6,color="#B45309",fontweight="bold")
    ax.text(50,16,"3 × 3 × 3 × 3 = 81 pipelines, antes de contar os hiperparâmetros.",ha="center",fontsize=9.5,color=INK,fontweight="bold")
    ax.text(50,8,"Escolher o tratamento dos dados é tão importante quanto escolher o algoritmo.",ha="center",fontsize=9,color=GRY,style="italic")
    save(fig,"f3_espaco")

if __name__=="__main__":
    for f in (f3_lavanderia,f3_onehot,f3_colunas,f3_vazamento,f3_escalas,f3_espaco): f()

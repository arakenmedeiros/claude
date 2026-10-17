from fc import *
from sklearn.datasets import make_circles, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def f8_materia():
    fig,ax=canvas(8.6,3.8)
    ax.text(50,97,"Atributos: a matéria-prima do modelo",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,52,24,34,"Dado bruto\n\ndata da compra\nvalor, cidade\nhora do dia",fc=GRY,ec=GRY,fs=8.5,tc="white",bold=True,r=2)
    arrow(ax,(27,69),(36,69),lw=2,ms=12); ax.text(31.5,73,"engenharia",ha="center",fontsize=8,color=GRY,style="italic")
    rbox(ax,37,52,28,34,"Atributos criados\n\ndia da semana\nvalor ÷ média do cliente\né fim de mês?\nlog(valor)",fc=IND,ec=IND,fs=8.5,tc="white",bold=True,r=2)
    arrow(ax,(66,69),(75,69),lw=2,ms=12); ax.text(70.5,73,"modelo",ha="center",fontsize=8,color=GRY,style="italic")
    rbox(ax,76,52,22,34,"Previsão\n\nmuito melhor\ncom bons\natributos",fc=GRN,ec=GRN,fs=8.5,tc="white",bold=True,r=2)
    ax.text(50,34,"\"Um algoritmo médio com bons atributos costuma vencer um algoritmo ótimo com atributos ruins.\"",ha="center",fontsize=9,color=INK,style="italic")
    ax.text(50,22,"Engenharia de atributos = transformar o que se mede em informação que o modelo consegue usar.",ha="center",fontsize=9,color=INK)
    ax.text(50,10,"Analogia: o chef que corta, tempera e combina os ingredientes antes de levar ao fogo.",ha="center",fontsize=8.5,color=GRY,style="italic")
    save(fig,"f8_materia")

def f8_circulos():
    X,y=make_circles(400,noise=0.08,factor=0.45,random_state=0)
    lin=make_pipeline(LogisticRegression()); a1=cross_val_score(lin,X,y,cv=5).mean()
    poly=make_pipeline(PolynomialFeatures(2,include_bias=False),LogisticRegression(max_iter=2000)); a2=cross_val_score(poly,X,y,cv=5).mean()
    r2=(X**2).sum(1)
    fig,axs=plt.subplots(1,3,figsize=(8.6,2.9))
    cols=np.where(y==0,IND,AMB)
    axs[0].scatter(X[:,0],X[:,1],c=cols,s=10,edgecolors="white",linewidths=0.3); axs[0].set_title(f"Atributos originais (x1, x2)\nreta não separa: {a1:.2f}",fontsize=8.8,fontweight="bold",color=RED)
    axs[0].set_xlabel("x1",fontsize=8); axs[0].set_ylabel("x2",fontsize=8)
    axs[1].scatter(X[:,0],r2,c=cols,s=10,edgecolors="white",linewidths=0.3); axs[1].axhline(0.45,c=GRN,lw=2); axs[1].set_title("Novo atributo: r² = x1² + x2²\numa reta já separa",fontsize=8.8,fontweight="bold",color=GRN)
    axs[1].set_xlabel("x1",fontsize=8); axs[1].set_ylabel("r²",fontsize=8)
    axs[2].bar(["originais","com termos\nquadráticos"],[a1,a2],color=[RED,GRN],width=0.55); axs[2].set_ylim(0,1.1); axs[2].set_title("Acurácia (validação cruzada)",fontsize=8.8,fontweight="bold")
    for i,v in enumerate([a1,a2]): axs[2].text(i,v+0.03,f"{v:.2f}",ha="center",fontweight="bold",fontsize=9)
    fig.tight_layout(); save(fig,"f8_circulos")

def f8_tres_familias():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"Três famílias de seleção de atributos",ha="center",fontsize=11,fontweight="bold")
    F=[("Filtros","avaliam cada atributo\nsozinho, antes do modelo\n(correlação, informação mútua)",IND,"rápidos e simples","SelectKBest"),
       ("Wrappers","testam subconjuntos\ntreinando o modelo\nde verdade (RFE)",CYA,"precisos; caros","RFE"),
       ("Embutidos","a seleção ocorre dentro do\ntreino (L1, importância\nde árvores)",GRN,"bom equilíbrio","L1 / árvores")]
    for i,(t,d,c,p,ex) in enumerate(F):
        x=2+i*33; rbox(ax,x,56,31,30,"",fc="white",ec=c,lw=2.2,r=2); ax.text(x+15.5,80,t,ha="center",fontsize=10.5,fontweight="bold",color=c); ax.text(x+15.5,67,d,ha="center",va="center",fontsize=8,color=INK,linespacing=1.2)
        rbox(ax,x+3,38,25,10,"Ponto forte:\n"+p,fc=c,ec=c,fs=8,tc="white",bold=True,r=1); ax.text(x+15.5,29,ex,ha="center",fontsize=8.8,color=c,family="monospace",fontweight="bold")
    ax.text(50,12,"Analogia: filtros = triagem por currículo; wrappers = entrevistar de verdade cada candidato; embutidos = avaliar durante o estágio.",ha="center",fontsize=8.3,color=GRY,style="italic")
    save(fig,"f8_tres_familias")

def f8_dfs():
    fig,ax=canvas(8.6,4.3)
    ax.text(50,97,"Síntese profunda de atributos (DFS): atributos a partir de tabelas relacionadas",ha="center",fontsize=10.5,fontweight="bold")
    rbox(ax,3,56,26,32,"Clientes\n\nid\nidade\ncidade",fc=IND,ec=IND,fs=8.5,tc="white",bold=True,r=2)
    rbox(ax,3,10,26,32,"Compras\n\nid_cliente\nvalor\ndata",fc=CYA,ec=CYA,fs=8.5,tc="white",bold=True,r=2)
    arrow(ax,(16,43),(16,55),lw=2,ms=12); ax.text(18,49,"1 cliente tem\nmuitas compras",fontsize=7.5,color=GRY)
    ax.text(38,70,"funções de\nagregação",ha="center",fontsize=9,color=INK,fontweight="bold",va="center")
    arrow(ax,(30,24),(46,40),lw=2,ms=12,c=GRY)
    rbox(ax,48,52,48,36,"Atributos criados automaticamente para cada cliente\n\nMEAN(compras.valor)\nMAX(compras.valor)\nSTD(compras.valor)\nCOUNT(compras)\nMEAN(compras.valor) / idade",fc="#EEF2FF",ec=IND,fs=8.3,tc=INK,lw=2,r=2)
    ax.text(72,38,"Profundidade 1: agregações diretas.\nProfundidade 2: agregações de agregações,\nconforme as tabelas se encadeiam.",ha="center",fontsize=8.3,color=INK,va="center")
    ax.text(50,5,"Cada função é aplicada ao longo dos caminhos entre tabelas, gerando centenas de atributos candidatos.",ha="center",fontsize=8.3,color=GRY,style="italic")
    save(fig,"f8_dfs")

def f8_explosao():
    fig,ax=plt.subplots(figsize=(8.0,3.5)); p=np.array([5,10,20,50,100])
    uni=8*p; par=6*p*(p-1)/2; ter=4*p*(p-1)*(p-2)/6
    ax.loglog(p,uni,"o-",c=IND,lw=2.2,label="1 atributo × 8 funções (log, raiz, quadrado...)")
    ax.loglog(p,par,"s-",c=AMB,lw=2.2,label="pares × 6 operações (soma, razão, produto...)")
    ax.loglog(p,ter,"^-",c=RED,lw=2.2,label="trios × 4 operações")
    ax.set_xlabel("número de atributos originais"); ax.set_ylabel("candidatos a criar (escala log)"); ax.legend(frameon=False,fontsize=8.2,loc="upper left")
    ax.set_title("A explosão de candidatos na criação automática de atributos",fontweight="bold",fontsize=10)
    save(fig,"f8_explosao")

def f8_loop():
    fig,ax=canvas(8.6,3.7)
    ax.text(50,97,"O ciclo da criação automática de atributos",ha="center",fontsize=11,fontweight="bold")
    B=[(2,"Gerar\ncandidatos\n(funções sobre\nos atributos)",IND),(21,"Filtrar\n(descartar o\nirrelevante e\no redundante)",CYA),(40,"Avaliar\n(validação\ncruzada, dentro\ndo pipeline)",GRN),(59,"Selecionar\no subconjunto\nque ajuda",AMB),(78,"Modelo\ncom atributos\nnovos",VIO)]
    for x,t,c in B: rbox(ax,x,45,19,36,t,fc=c,ec=c,fs=8.3,tc="white",bold=True,r=2)
    for x in (21.2,40.2,59.2,78.2): arrow(ax,(x-1.4,63),(x-0.2,63),lw=1.6,ms=8)
    rbox(ax,3,6,94,24,"Tudo isso deve acontecer DENTRO da validação cruzada: gerar e selecionar atributos\nolhando os rótulos de todos os dados é vazamento (Capítulo 3).\nCom muitos candidatos, a chance de um atributo parecer bom só por sorte cresce:\no overfitting da busca (Capítulo 4) também se aplica aqui.",fc="#FEF3C7",ec=AMB,fs=8.2,tc=INK,lw=1.8,r=1.5)
    save(fig,"f8_loop")

if __name__=="__main__":
    for fn_ in (f8_materia,f8_circulos,f8_tres_familias,f8_dfs,f8_explosao,f8_loop): fn_()

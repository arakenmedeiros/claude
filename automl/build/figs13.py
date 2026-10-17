from fc import *
import json

def f13_perguntas():
    fig,ax=canvas(8.6,3.8)
    ax.text(50,97,"Três perguntas antes de confiar em um modelo",ha="center",fontsize=11,fontweight="bold")
    Q=[("1","Eu entendo\no que ele faz?","explicabilidade",IND),("2","Ele trata as\npessoas com justiça?","ética e equidade",GRN),("3","Quanto custa e o que\nele não consegue?","custo e limites",AMB)]
    for i,(n,q,t,c) in enumerate(Q):
        x=2+i*33; rbox(ax,x,22,30,62,"",fc="white",ec=c,lw=2.4,r=2)
        ax.add_patch(Circle((x+15,70),6,fc=c,ec=c)); ax.text(x+15,70,n,ha="center",va="center",color="white",fontsize=14,fontweight="bold")
        ax.text(x+15,50,q,ha="center",va="center",fontsize=9.2,color=INK,linespacing=1.4)
        rbox(ax,x+3,27,24,10,t,fc=c,ec=c,fs=8.2,tc="white",bold=True,r=1.5)
    ax.text(50,9,"O AutoML desloca a responsabilidade humana, não a elimina.",ha="center",fontsize=8.8,color=GRY,style="italic")
    save(fig,"f13_perguntas")

def f13_explicacao():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"Explicação global e explicação local",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,12,47,76,"",fc="white",ec=IND,lw=2.2,r=2); ax.text(25.5,80,"Global",ha="center",fontsize=10,fontweight="bold",color=IND)
    ax.text(25.5,70,"O que o modelo valoriza, em geral?",ha="center",fontsize=8.4,color=INK)
    for i,(nm,v) in enumerate([("renda",1.0),("emprego",0.45),("score",0.4),("dívida",0.38)]):
        y=56-i*10; ax.text(8,y,nm,fontsize=8,va="center"); ax.add_patch(Rectangle((20,y-2.6),v*24,5.2,fc=IND,ec="none"))
    ax.text(25.5,17,"importância média dos atributos",ha="center",fontsize=7.8,color=GRY,style="italic")
    rbox(ax,51,12,47,76,"",fc="white",ec=GRN,lw=2.2,r=2); ax.text(74.5,80,"Local",ha="center",fontsize=10,fontweight="bold",color=GRN)
    ax.text(74.5,70,"Por que esta pessoa foi negada?",ha="center",fontsize=8.4,color=INK)
    for i,(nm,v) in enumerate([("renda",-1.0),("emprego",-0.78),("score",-0.5),("dívida",-0.27)]):
        y=56-i*10; ax.text(56,y,nm,fontsize=8,va="center"); ax.add_patch(Rectangle((82,y-2.6),v*14,5.2,fc=RED,ec="none"))
    ax.plot([82,82],[18,62],color=INK,lw=1); ax.text(74.5,17,"contribuição de cada atributo (SHAP)",ha="center",fontsize=7.8,color=GRY,style="italic")
    save(fig,"f13_explicacao")

def f13_importancia():
    d=json.load(open("/home/user/claude/automl/build/f13_expl.json")); nm=d["nomes"]
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.2),gridspec_kw={"width_ratios":[1.1,1]})
    o=np.argsort(d["shap"]); a=axs[0]; y=np.arange(len(nm))
    sh=np.array(d["shap"])/max(d["shap"]); pm=np.array(d["perm"])/max(d["perm"])
    a.barh(y+.2,sh[o],.38,color=IND,label="SHAP (normalizado)"); a.barh(y-.2,pm[o],.38,color=AMB,label="permutação (normalizada)")
    a.set_yticks(y); a.set_yticklabels([nm[i] for i in o],fontsize=8); a.set_title("Importância global",fontsize=9.5); a.legend(frameon=False,fontsize=7,loc="lower right")
    b=axs[1]; loc=np.array(d["local"]); o2=np.argsort(np.abs(loc)); b.barh(np.arange(len(nm)),loc[o2],color=[RED if v<0 else GRN for v in loc[o2]])
    b.set_yticks(np.arange(len(nm))); b.set_yticklabels([f"{nm[i]} = {d['valores'][i]:.1f}" for i in o2],fontsize=7.6); b.axvline(0,color=INK,lw=1)
    b.set_title(f"Explicação local (P(aprovar) = {d['p']:.2f})",fontsize=9.5); b.set_xlabel("contribuição SHAP",fontsize=8)
    fig.tight_layout(); save(fig,"f13_importancia")

def f13_metricas():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"Duas formas de comparar grupos",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,18,47,68,"",fc="white",ec=IND,lw=2.2,r=2); ax.text(25.5,78,"Taxa de seleção",ha="center",fontsize=10,fontweight="bold",color=IND)
    ax.text(25.5,64,"de todas as pessoas do grupo,\nquantas foram aprovadas?",ha="center",fontsize=8.4,color=INK,linespacing=1.35)
    ax.text(25.5,38,"selecionados do grupo\n─────────────────\n todos do grupo",ha="center",fontsize=8.6,color=IND,family="monospace")
    rbox(ax,51,18,47,68,"",fc="white",ec=GRN,lw=2.2,r=2); ax.text(74.5,78,"Igualdade de oportunidade",ha="center",fontsize=10,fontweight="bold",color=GRN)
    ax.text(74.5,64,"entre quem MERECERIA a aprovação,\nquantos foram aprovados? (TPR)",ha="center",fontsize=8.4,color=INK,linespacing=1.35)
    ax.text(74.5,38,"merecedores aprovados\n─────────────────\n  todos os merecedores",ha="center",fontsize=8.6,color=GRN,family="monospace")
    ax.text(50,8,"Iguais entre os grupos? Se não, há disparidade a investigar.",ha="center",fontsize=8.6,color=GRY,style="italic")
    save(fig,"f13_metricas")

def f13_proxy():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"Apagar a coluna \"grupo\" não apaga o viés",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,60,20,24,"Grupo\n(removido)",fc=GRY,ec=GRY,fs=8.8,tc="white",bold=True,r=2)
    P=[("renda",76),("CEP-score",60),("emprego",44)]
    for nm,y in P:
        rbox(ax,38,y-5,20,11,nm,fc=AMB,ec=AMB,fs=8.4,tc="white",bold=True,r=1.5)
        arrow(ax,(22.5,70),(37.5,y+0.5),c=RED,lw=1.6,ms=9)
        arrow(ax,(58.5,y+0.5),(72.5,64),c=GRY,lw=1.4,ms=9)
    rbox(ax,73,48,25,34,"Modelo\n(aprende com\nrótulos\nenviesados)",fc=IND,ec=IND,fs=8.2,tc="white",bold=True,r=2)
    rbox(ax,2,4,96,16,"Variáveis correlacionadas (proxies) e rótulos históricos carregam o viés para dentro do modelo.",fc="#FEF3C7",ec=AMB,fs=8.8,tc=INK,lw=1.8,r=2)
    save(fig,"f13_proxy")

def f13_justica():
    d=json.load(open("/home/user/claude/automl/build/f13_just.json"))
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.2))
    cfg=[("limiar\núnico","base"),("sem o\nCEP","sem_cep"),("limiares\npor grupo","ajust")]
    for a,(chave,tit) in zip(axs,(("sel","Taxa de seleção"),("tpr","TPR real (entre quem pagaria)"))):
        x=np.arange(3); w=.36
        for k,(c,lab) in enumerate(((IND,"grupo 0"),(AMB,"grupo 1"))):
            v=[d[ck][chave][k] for _,ck in cfg]; bars=a.bar(x+(k-.5)*w,v,w,color=c,label=lab)
            for b in bars: a.text(b.get_x()+b.get_width()/2,b.get_height()+.01,f"{b.get_height():.2f}",ha="center",fontsize=7.5)
        a.set_xticks(x); a.set_xticklabels([c for c,_ in cfg],fontsize=8); a.set_ylim(0,1.0); a.set_title(tit,fontsize=9.5); a.legend(frameon=False,fontsize=7.5,loc="upper left",ncol=2)
    fig.tight_layout(); save(fig,"f13_justica")

def f13_limites():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"O que o AutoML faz e o que continua humano",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,14,47,72,"",fc="#ECFDF5",ec=GRN,lw=2.2,r=2); ax.text(25.5,78,"O AutoML faz",ha="center",fontsize=10,fontweight="bold",color=GRN)
    ax.text(25.5,48,"• busca algoritmos e ajustes\n• compara com validação\n• combina modelos\n• economiza tempo\n• padroniza o processo",ha="center",va="center",fontsize=8.6,color=INK,linespacing=1.7)
    rbox(ax,51,14,47,72,"",fc="#FEF2F2",ec=RED,lw=2.2,r=2); ax.text(74.5,78,"Continua humano",ha="center",fontsize=10,fontweight="bold",color=RED)
    ax.text(74.5,48,"• definir o problema e a métrica\n• garantir dados de qualidade\n• checar justiça e vieses\n• explicar e auditar\n• monitorar depois da implantação",ha="center",va="center",fontsize=8.6,color=INK,linespacing=1.7)
    save(fig,"f13_limites")

def f13_checklist():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"Lista de verificação ética",ha="center",fontsize=11,fontweight="bold")
    C=[("ANTES",IND,"• Quem é afetado?\n• De onde vêm os dados?\n• Quem está ausente?\n• Os rótulos são a verdade\n  ou decisões passadas?"),("DURANTE",GRN,"• Há linha de base e modelo\n  interpretável?\n• A validação respeita a\n  estrutura dos dados?\n• Medi o desempenho por grupo?"),("DEPOIS",AMB,"• Há explicação adequada?\n• Há canal de contestação?\n• Há monitoramento de\n  deriva e disparidade?\n• O custo foi registrado?")]
    for i,(t,c,d) in enumerate(C):
        x=2+i*33; rbox(ax,x,70,30,16,t,fc=c,ec=c,fs=10,tc="white",bold=True,r=2); rbox(ax,x,8,30,58,"",fc="white",ec=c,lw=2.2,r=2)
        ax.text(x+2,37,d,ha="left",va="center",fontsize=7.9,color=INK,linespacing=1.55)
    save(fig,"f13_checklist")

if __name__=="__main__":
    for fn_ in (f13_perguntas,f13_explicacao,f13_importancia,f13_metricas,f13_proxy,f13_justica,f13_limites,f13_checklist): fn_()

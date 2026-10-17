from fc import *
import json

def f10_feira():
    fig,ax=canvas(8.6,4.4)
    ax.text(50,97,"A feira de ferramentas de AutoML para dados tabulares",ha="center",fontsize=11,fontweight="bold")
    T=[("Auto-sklearn","meta-aprendizado\n+ otimização\nbayesiana +\nensemble",IND,"Feurer et al."),("TPOT","programação\ngenética evolui\npipelines inteiros",GRN,"Olson e Moore"),("H2O AutoML","muitos modelos +\nensembles\nempilhados, em\nplataforma\ndistribuída",AMB,"LeDell e Poirier"),("AutoGluon","stacking em\ncamadas, ênfase\nem robustez e\nprecisão",VIO,"Erickson et al."),("FLAML","busca de baixo\ncusto: leve e rápida",CYA,"Wang et al.")]
    w=18.4
    for i,(n,d,c,a) in enumerate(T):
        x=1+i*19.6
        rbox(ax,x,60,w,28,n,fc=c,ec=c,fs=9.5,tc="white",bold=True,r=2)
        rbox(ax,x,22,w,34,d,fc="white",ec=c,fs=6.9,tc=INK,lw=2,r=2)
        ax.text(x+w/2,13,a,ha="center",fontsize=7.6,color=GRY,style="italic")
    ax.text(50,3,"Todas seguem o mesmo contrato: dados + orçamento entram, modelo pronto sai.",ha="center",fontsize=8.6,color=INK)
    save(fig,"f10_feira")

def f10_roteiro():
    fig,ax=canvas(8.6,3.4)
    ax.text(50,96,"O roteiro comum das ferramentas de AutoML",ha="center",fontsize=11,fontweight="bold")
    S=[("1. Você entrega\ndados + orçamento",GRY),("2. A ferramenta\nescolhe tratamento,\nalgoritmos e ajustes",IND),("3. Avalia cada\ncandidato por\nvalidação",CYA),("4. Devolve o melhor\nmodelo ou um\nensemble",GRN)]
    for i,(t,c) in enumerate(S):
        x=2+i*24.5
        rbox(ax,x,38,21,38,t,fc=c,ec=c,fs=8.6,tc="white",bold=True,r=2)
        if i<3: arrow(ax,(x+21.4,57),(x+24.1,57),lw=2,ms=11)
    arrow(ax,(86,36),(86,24),c=GRY,lw=1.5,ms=9); arrow(ax,(86,24),(12,24),c=GRY,lw=1.5,ms=1); arrow(ax,(12,24),(12,37),c=GRY,lw=1.5,ms=9)
    ax.text(50,14,"repete até o tempo (orçamento) acabar",ha="center",fontsize=8.8,color=GRY,style="italic")
    save(fig,"f10_roteiro")

def f10_botoes():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"Os botões de uma chamada de AutoML",ha="center",fontsize=11,fontweight="bold")
    B=[("time_budget","quanto tempo\nbuscar",IND),("metric","o que\notimizar",CYA),("estimator_list","quais algoritmos\npodem entrar",GRN),("n_splits","quão estável\né a validação",AMB),("seed","fixa o\nacaso",VIO),("max_iter","limita as\ntentativas",PNK)]
    for i,(n,d,c) in enumerate(B):
        x=2+(i%3)*33; y=52 if i<3 else 8
        rbox(ax,x,y,30,38,"",fc="white",ec=c,lw=2.2,r=2)
        ax.text(x+15,y+29,n,ha="center",fontsize=9.2,fontweight="bold",color=c,family="monospace")
        ax.text(x+15,y+13,d,ha="center",va="center",fontsize=8.6,color=INK,linespacing=1.3)
    save(fig,"f10_botoes")

def f10_orcamento():
    d=json.load(open("/home/user/claude/automl/build/f10_orcamento.json"))
    ks=sorted(d,key=int); fig,ax=plt.subplots(figsize=(6.6,3.4))
    for j,k in enumerate(ks):
        for v in d[k]: ax.scatter(int(k),v,s=46,color=CL[1],zorder=3,alpha=.8)
    m=[np.mean(d[k]) for k in ks]; ax.plot([int(k) for k in ks],m,"-o",color=IND,lw=2.4,ms=7,label="média das duas divisões",zorder=4)
    for k,v in zip(ks,m): ax.text(int(k)+0.5,v-0.007,f"{v:.3f}",ha="center",fontsize=8,fontweight="bold",color=IND)
    ax.set_xlabel("orçamento de tempo (segundos)"); ax.set_ylabel("acurácia no teste"); ax.set_xticks([int(k) for k in ks])
    lo=min(min(v) for v in d.values()); hi=max(max(v) for v in d.values()); ax.set_ylim(lo-0.02,hi+0.02)
    ax.set_title("Mais tempo ajuda no começo; depois, retornos decrescentes",fontsize=10); ax.legend(frameon=False,fontsize=8,loc="lower right"); ax.grid(axis="y",color=MID,lw=.6)
    save(fig,"f10_orcamento")

def f10_justa():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"Seis cuidados para uma comparação justa",ha="center",fontsize=11,fontweight="bold")
    C=[("1","mesmos dados e\nmesmas divisões",IND),("2","mesmo orçamento\n(tempo ou tentativas)",CYA),("3","teste final intocado,\nvisto uma só vez",GRN),("4","várias divisões\npara ver a variação",AMB),("5","várias bases\nde dados",VIO),("6","linhas de base\nsimples",PNK)]
    for i,(n,t,c) in enumerate(C):
        x=2+(i%3)*33; y=52 if i<3 else 8
        rbox(ax,x,y,30,38,"",fc="white",ec=c,lw=2.2,r=2)
        ax.add_patch(Circle((x+5.5,y+31),4.2,fc=c,ec=c)); ax.text(x+5.5,y+31,n,ha="center",va="center",color="white",fontsize=10,fontweight="bold")
        ax.text(x+17,y+17,t,ha="center",va="center",fontsize=8.6,color=INK,linespacing=1.35)
    save(fig,"f10_justa")

def f10_bench():
    d=json.load(open("/home/user/claude/automl/build/f10_cache.json")); res=d["res"]; ms=d["metodos"]; bs=list(res)
    fig,axs=plt.subplots(1,len(bs),figsize=(8.6,3.4))
    for a,b in zip(axs,bs):
        for j,m in enumerate(ms):
            v=res[b][m]; a.scatter([j]*len(v),v,s=22,color=CL[j],alpha=.55,zorder=3); a.plot([j-.3,j+.3],[np.mean(v)]*2,color=CL[j],lw=3,zorder=4)
        a.set_title(b,fontsize=9.5); a.set_xticks(range(len(ms))); a.set_xticklabels(["Flor.","Boost.","Busca","FLAML"],fontsize=6.3,rotation=0); a.grid(axis="y",color=MID,lw=.6)
        allv=[x for m in ms for x in res[b][m]]; a.set_ylim(min(allv)-0.015,max(allv)+0.015)
    axs[0].set_ylabel("acurácia no teste")
    fig.suptitle("Quatro métodos em quatro bases (pontos = divisões; traço = média)",fontsize=9.6,fontweight="bold",y=1.03); fig.tight_layout(); save(fig,"f10_bench")

def f10_decisao():
    fig,ax=canvas(8.6,4.4)
    ax.text(50,97,"Qual ferramenta escolher? Um roteiro de perguntas",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,36,76,28,14,"Qual é a sua\nprioridade?",fc=NAVY,ec=NAVY,fs=9,tc="white",bold=True,r=2)
    O=[(1,"Rapidez e\npouco recurso","FLAML",CYA),(21,"Entender e\nauditar o pipeline","TPOT",GRN),(41,"Máxima precisão\n(tempo e memória)","AutoGluon\nou H2O",VIO),(61,"Ecossistema\nscikit-learn","Auto-sklearn",IND),(81,"Dados muito\ngrandes / cluster","H2O",AMB)]
    for x,q,r,c in O:
        arrow(ax,(50,75.5),(x+9,56),c=GRY,lw=1.4,ms=9)
        rbox(ax,x,38,18,17,q,fc="white",ec=c,fs=7.6,tc=INK,lw=2,r=2)
        arrow(ax,(x+9,37.5),(x+9,29),c=c,lw=1.6,ms=9)
        rbox(ax,x,11,18,17,r,fc=c,ec=c,fs=8.6,tc="white",bold=True,r=2)
    ax.text(50,3,"Em dúvida, compare duas ou três ferramentas com o mesmo orçamento e a mesma validação.",ha="center",fontsize=8.3,color=GRY,style="italic")
    save(fig,"f10_decisao")

if __name__=="__main__":
    for fn_ in (f10_feira,f10_roteiro,f10_botoes,f10_orcamento,f10_justa,f10_bench,f10_decisao):
        try: fn_()
        except FileNotFoundError as e: print("pendente",fn_.__name__,e)

"""SHIM DE TESTE (nao faz parte do livro): imita ucimlrepo.fetch_ucirepo com dados sinteticos
com as mesmas colunas, para validar a logica dos codigos sem acesso a internet."""
import numpy as np, pandas as pd
from types import SimpleNamespace as NS
def fetch_ucirepo(name=None, id=None):
    rng=np.random.default_rng(0)
    if id==73:   # Mushroom
        n=4000
        cols={"cap-shape":"bcxfks","cap-surface":"fgys","cap-color":"nbcgrpuewy","bruises":"tf","odor":"almnpcyfs","gill-attachment":"adfn","gill-spacing":"cwd","gill-size":"bn","gill-color":"knbhgropuewy","stalk-shape":"et","stalk-root":"bcuezr","stalk-surface-above-ring":"fyks","stalk-surface-below-ring":"fyks","stalk-color-above-ring":"nbcgopewy","stalk-color-below-ring":"nbcgopewy","veil-type":"p","veil-color":"nwoy","ring-number":"not","ring-type":"ceflnpsz","spore-print-color":"knbhrouwy","population":"acnsvy","habitat":"glmpuwd"}
        X=pd.DataFrame({k:rng.choice(list(v),n) for k,v in cols.items()})
        X.loc[rng.random(n)<0.3,"stalk-root"]=np.nan
        pois=((X["odor"].isin(list("fcyspm")))|((X["spore-print-color"]=="h")&(X["gill-size"]=="n")))
        flip=rng.random(n)<0.01
        y=pd.DataFrame({"poisonous":np.where(pois^flip,"p","e")})
        return NS(data=NS(features=X,targets=y,original=pd.concat([X,y],axis=1)))
    if id==222:  # Bank Marketing
        n=6000
        X=pd.DataFrame({"age":rng.integers(18,90,n),"job":rng.choice(["admin.","blue-collar","technician","services","management","retired","student"],n),
         "marital":rng.choice(["married","single","divorced"],n),"education":rng.choice(["primary","secondary","tertiary"],n),
         "default":rng.choice(["no","yes"],n,p=[.98,.02]),"balance":rng.normal(1300,3000,n).round(),"housing":rng.choice(["yes","no"],n),
         "loan":rng.choice(["yes","no"],n,p=[.16,.84]),"contact":rng.choice(["cellular","telephone",np.nan],n,p=[.6,.1,.3]),
         "day_of_week":rng.integers(1,29,n),"month":rng.choice(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],n),
         "duration":rng.exponential(250,n).round(),"campaign":rng.integers(1,10,n),"pdays":np.where(rng.random(n)<.8,-1,rng.integers(1,400,n)),
         "previous":rng.integers(0,5,n),"poutcome":rng.choice(["failure","success","other",np.nan],n,p=[.1,.04,.04,.82])})
        logit=-2.6+0.004*(X["duration"])+0.0002*X["balance"]+1.2*(X["poutcome"]=="success")-0.5*(X["housing"]=="yes")+0.6*(X["job"].isin(["student","retired"]))
        p=1/(1+np.exp(-logit)); y=pd.DataFrame({"y":np.where(rng.random(n)<p,"yes","no")})
        return NS(data=NS(features=X,targets=y,original=pd.concat([X,y],axis=1)))
    if id==186:  # Wine quality
        n=3000
        X=pd.DataFrame({"fixed_acidity":rng.normal(7.2,1.3,n),"volatile_acidity":rng.gamma(4,.08,n),"citric_acid":rng.uniform(0,.9,n),"residual_sugar":rng.gamma(2,2.5,n),
         "chlorides":rng.gamma(3,.02,n),"free_sulfur_dioxide":rng.gamma(4,8,n),"total_sulfur_dioxide":rng.gamma(5,25,n),"density":rng.normal(.995,.003,n),
         "pH":rng.normal(3.2,.15,n),"sulphates":rng.gamma(6,.09,n),"alcohol":rng.normal(10.5,1.2,n)})
        q=np.clip(np.round(3+0.45*(X["alcohol"]-9)-2.0*X["volatile_acidity"]+rng.normal(0,.7,n)),3,9).astype(int)
        y=pd.DataFrame({"quality":q}); orig=pd.concat([X,y],axis=1); orig["color"]=rng.choice(["red","white"],n)
        return NS(data=NS(features=X,targets=y,original=orig))
    if id==352:  # Online Retail
        n=20000
        cust=rng.integers(12000,12800,n).astype(float); cust[rng.random(n)<.1]=np.nan
        d=pd.to_datetime("2010-12-01")+pd.to_timedelta(rng.integers(0,370*24*60,n),unit="m")
        X=pd.DataFrame({"InvoiceNo":rng.integers(536000,580000,n).astype(str),"StockCode":rng.integers(1000,1100,n).astype(str),"Description":"ITEM",
          "Quantity":rng.integers(1,12,n),"InvoiceDate":d,"UnitPrice":rng.gamma(2,2,n).round(2),"CustomerID":cust,"Country":rng.choice(["United Kingdom","France","Germany"],n,p=[.9,.05,.05])})
        return NS(data=NS(features=X,targets=None,original=X))
    raise ValueError(id)

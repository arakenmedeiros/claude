import warnings; warnings.filterwarnings("ignore")
import numpy as np, time
from flaml import AutoML
from scipy.stats import loguniform, randint
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
t0=time.time()
X,y=load_digits(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=0)
t=time.time(); m=HistGradientBoostingClassifier(random_state=0).fit(Xtr,ytr); print("hgb",time.time()-t,m.score(Xte,yte))
t=time.time(); m=RandomForestClassifier(200,random_state=0,n_jobs=-1).fit(Xtr,ytr); print("rf",time.time()-t,m.score(Xte,yte))
t=time.time()
r=RandomizedSearchCV(HistGradientBoostingClassifier(random_state=0,max_iter=60),{"learning_rate":loguniform(.01,.5),"max_leaf_nodes":randint(4,40),"min_samples_leaf":randint(5,60),"l2_regularization":loguniform(1e-4,10)},n_iter=10,cv=3,random_state=0,n_jobs=-1).fit(Xtr,ytr); print("rs",time.time()-t,r.score(Xte,yte))

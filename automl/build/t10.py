import warnings; warnings.filterwarnings("ignore")
import numpy as np, time
from flaml import AutoML
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X,y=load_breast_cancer(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=0)
a=AutoML(); t=time.time()
a.fit(Xtr,ytr,task="classification",time_budget=8,metric="accuracy",seed=0,verbose=0,estimator_list=["lgbm","rf","extra_tree","lrl1"],eval_method="cv",n_splits=3)
print(time.time()-t,a.best_estimator,a.best_config,a.best_loss,accuracy_score(yte,a.predict(Xte)))
print(a.best_config_per_estimator.keys(), len(a.config_history))
print(list(a.config_history.items())[:3])
print(a.model.estimator)

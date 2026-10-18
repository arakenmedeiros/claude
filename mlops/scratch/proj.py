import json, hashlib, os, subprocess, sys, tempfile, textwrap, warnings
import joblib, numpy as np, pandas as pd
import mlflow, mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from scipy.stats import loguniform

warnings.filterwarnings("ignore")
os.chdir(tempfile.mkdtemp())
SEMENTE = 42
rng = np.random.default_rng(SEMENTE)

# ------------------------------------------------------------------ 1. dados, validação e versão
X, y = load_digits(return_X_y=True)
def validar(X, y):
    erros = []
    if X.shape[1] != 64: erros.append("número de colunas diferente de 64")
    if np.isnan(X).any(): erros.append("há valores ausentes")
    if X.min() < 0 or X.max() > 16: erros.append("pixels fora de [0, 16]")
    if set(np.unique(y)) != set(range(10)): erros.append("classes diferentes de 0..9")
    return erros
print("[1] validação:", validar(X, y) or "aprovada")
hash_dados = hashlib.sha256(np.c_[X, y].tobytes()).hexdigest()[:12]
print("[1] versão dos dados (hash):", hash_dados)

# 60% desenvolvimento, 40% "futuro" (produção simulada, 8 semanas)
Xdev, Xfut, ydev, yfut = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEMENTE)
Xtr, Xte, ytr, yte = train_test_split(Xdev, ydev, test_size=0.25, stratify=ydev, random_state=SEMENTE)
print(f"[1] treino={len(Xtr)} teste={len(Xte)} produção simulada={len(Xfut)}")

# ------------------------------------------------------------------ 2/3. candidatos rastreados no MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("digitos")
cv = StratifiedKFold(5, shuffle=True, random_state=SEMENTE)
candidatos = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logreg": LogisticRegression(max_iter=5000),
    "floresta": RandomForestClassifier(n_estimators=200, random_state=SEMENTE),
    "svc": RandomizedSearchCV(SVC(),
                              {"C": loguniform(0.1, 100), "gamma": loguniform(1e-4, 1e-1)},
                              n_iter=12, cv=cv, scoring="f1_macro", random_state=SEMENTE),
}
resultados = {}
for nome, modelo in candidatos.items():
    with mlflow.start_run(run_name=nome):
        modelo.fit(Xtr, ytr)
        est = modelo.best_estimator_ if hasattr(modelo, "best_estimator_") else modelo
        f1_cv = cross_val_score(est, Xtr, ytr, cv=cv, scoring="f1_macro").mean()
        mlflow.log_params({"modelo": nome, "semente": SEMENTE})
        mlflow.log_metric("f1_macro_cv", f1_cv)
        mlflow.set_tag("hash_dados", hash_dados)
        info = mlflow.sklearn.log_model(est, name="modelo", signature=infer_signature(Xte, est.predict(Xte)),
                                        serialization_format="cloudpickle")
        resultados[nome] = (f1_cv, est, info.model_uri)
for nome, (f1_cv, _, _) in resultados.items():
    print(f"[2] {nome:9s} F1-macro (validação cruzada) = {f1_cv:.3f}")

# ------------------------------------------------------------------ 3. portão de qualidade e registro (apelido champion)
PISO_F1, MARGEM = 0.95, 0.01
melhor = max((k for k in resultados if k != "baseline"), key=lambda k: resultados[k][0])
f1_melhor, modelo_melhor, uri_melhor = resultados[melhor]
f1_base = resultados["baseline"][0]
aprovado = f1_melhor >= PISO_F1 and f1_melhor >= f1_base + MARGEM
print(f"[3] melhor candidato = {melhor} (F1 {f1_melhor:.3f}) | piso {PISO_F1} | portão {'APROVADO' if aprovado else 'REPROVADO'}")
assert aprovado
cliente = MlflowClient()
NOME = "digitos"
v = mlflow.register_model(uri_melhor, NOME)
cliente.set_registered_model_alias(NOME, "champion", v.version)
champion = mlflow.sklearn.load_model(f"models:/{NOME}@champion")
f1_teste = f1_score(yte, champion.predict(Xte), average="macro")
print(f"[3] versão {v.version} registrada como champion | F1-macro no teste (uso único) = {f1_teste:.3f}")

# ------------------------------------------------------------------ 4. API com contrato, carregando o champion pelo apelido
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, field_validator

class Imagem(BaseModel):
    pixels: list[float] = Field(min_length=64, max_length=64)
    @field_validator("pixels")
    @classmethod
    def faixa(cls, v):
        if min(v) < 0 or max(v) > 16:
            raise ValueError("pixels devem estar entre 0 e 16")
        return v

def criar_app():
    estado = {}
    @asynccontextmanager
    async def ciclo(app):
        estado["m"] = mlflow.sklearn.load_model(f"models:/{NOME}@champion")
        estado["v"] = cliente.get_model_version_by_alias(NOME, "champion").version
        yield
    app = FastAPI(lifespan=ciclo)
    @app.post("/previsao")
    def prever(img: Imagem):
        x = np.array(img.pixels).reshape(1, -1)
        return {"digito": int(estado["m"].predict(x)[0]), "versao_modelo": estado["v"]}
    return app

with TestClient(criar_app()) as api:
    ok = api.post("/previsao", json={"pixels": Xte[0].tolist()})
    ruim = api.post("/previsao", json={"pixels": [99.0] * 64})
    curta = api.post("/previsao", json={"pixels": [1.0] * 10})
print(f"[4] API: válida -> {ok.status_code} {ok.json()} (real = {yte[0]}) | pixel fora da faixa -> {ruim.status_code} | tamanho errado -> {curta.status_code}")

# ------------------------------------------------------------------ 5. testes automáticos (pytest) sobre o artefato exportado
joblib.dump(champion, "champion.joblib")
np.savez("teste.npz", X=Xte, y=yte)
open("test_modelo.py", "w").write(textwrap.dedent('''
    import joblib, numpy as np
    from sklearn.metrics import f1_score
    M = joblib.load("champion.joblib"); D = np.load("teste.npz"); X, y = D["X"], D["y"]

    def test_desempenho_minimo():
        assert f1_score(y, M.predict(X), average="macro") >= 0.95

    def test_invariancia_a_ruido_minimo():
        ruido = np.random.default_rng(0).normal(0, 0.2, X.shape)
        mudou = (M.predict(X) != M.predict(np.clip(X + ruido, 0, 16))).mean()
        assert mudou <= 0.05

    def test_todas_as_classes_sao_reconhecidas():
        rec = [(M.predict(X[y == c]) == c).mean() for c in range(10)]
        assert min(rec) >= 0.85

    def test_determinismo():
        assert np.array_equal(M.predict(X), M.predict(X))
'''))
r = subprocess.run([sys.executable, "-m", "pytest", "test_modelo.py", "-q", "-p", "no:cacheprovider"], capture_output=True, text=True)
print("[5] pytest:", r.stdout.strip().splitlines()[-1])

# ------------------------------------------------------------------ 6/7. produção simulada com deriva, monitoramento e CT
def psi(ref, atual, faixas=10):
    cortes = np.quantile(ref, np.linspace(0, 1, faixas + 1)); cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.clip(np.histogram(ref, cortes)[0] / len(ref), 1e-6, None)
    q = np.clip(np.histogram(atual, cortes)[0] / len(atual), 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))

def auc_dominio(A, B):
    Z = np.vstack([A, B]); d = np.r_[np.zeros(len(A)), np.ones(len(B))]
    return cross_val_score(RandomForestClassifier(100, min_samples_leaf=3, random_state=0, n_jobs=2), Z, d, cv=4, scoring="roc_auc").mean()

from scipy.ndimage import gaussian_filter
def aplicar_deriva(X, semana, r):          # a câmera sai de foco aos poucos e a tinta desbota
    imgs = X.reshape(-1, 8, 8)
    nivel = min(semana, 4)                       # a degradação cresce por 4 semanas e depois se estabiliza
    emb = np.stack([gaussian_filter(im, sigma=FOCO * nivel, mode="constant") for im in imgs])
    return np.clip(emb.reshape(len(X), -1) * (1 - FADE * nivel), 0, 16)

FADE, FOCO = 0.03, 0.30                      # intensidade da degradação por semana
SEMANAS = 8
lotes = np.array_split(np.arange(len(Xfut)), SEMANAS)
rd = np.random.default_rng(7)
prod = [(aplicar_deriva(Xfut[i], s + 1, rd), yfut[i]) for s, i in enumerate(lotes)]

ref_int = Xtr.std(axis=1)
ref_amostra = Xtr[rng.choice(len(Xtr), 300, replace=False)]
fixo = champion                              # v1 congelado (sem CT), para comparação
atual = champion                             # champion corrente (com CT)
versao_atual = int(v.version)
rotulados_X, rotulados_y = [], []            # lotes já rotulados (chegam com 1 semana de atraso)
linhas = []
for s, (Xs, ys) in enumerate(prod, start=1):
    p_atual = atual.predict(Xs)
    conf = atual.decision_function(Xs).max(axis=1).mean() if hasattr(atual, "decision_function") else np.nan
    m = {"semana": s,
         "PSI_contraste": psi(ref_int, Xs.std(axis=1)),
         "AUC_dominio": auc_dominio(ref_amostra, Xs),
         "acc_sem_CT": (fixo.predict(Xs) == ys).mean(),
         "acc_com_CT": (p_atual == ys).mean(),
         "versao": versao_atual, "acao": ""}
    alerta = m["PSI_contraste"] > 0.25 or m["AUC_dominio"] > 0.80
    # CT: rótulos da semana s-1 já chegaram; a semana s-2 e anteriores entram no treino
    if alerta and s >= 3:
        Xtreino = np.vstack([Xtr] + [a for a in rotulados_X[:-1]]); ytreino = np.concatenate([ytr] + rotulados_y[:-1])
        pesos = np.r_[np.ones(len(Xtr)), np.concatenate([np.full(len(b), 3.0) for b in rotulados_y[:-1]])] if rotulados_y[:-1] else np.ones(len(Xtreino))
        cand = SVC(C=modelo_melhor.C, gamma=modelo_melhor.gamma)
        cand.fit(Xtreino, ytreino, sample_weight=pesos)
        Xv, yv = rotulados_X[-1], rotulados_y[-1]          # semana s-1: recente e não vista pelo candidato
        f1_c, f1_a = f1_score(yv, cand.predict(Xv), average="macro"), f1_score(yv, atual.predict(Xv), average="macro")
        if f1_c >= f1_a + 0.02:
            nv = mlflow.register_model(mlflow.sklearn.log_model(cand, name="modelo_ct").model_uri, NOME)
            cliente.set_registered_model_alias(NOME, "champion", nv.version)
            atual, versao_atual = cand, int(nv.version)
            ref_int = np.concatenate([b.std(axis=1) for b in rotulados_X[-2:]])          # nova referência: dados recentes
            ref_amostra = np.vstack(rotulados_X[-2:])
            m["acao"] = f"promoveu v{nv.version} (F1 recente {f1_a:.2f}->{f1_c:.2f})"
        else:
            m["acao"] = f"manteve (candidato {f1_c:.2f} vs {f1_a:.2f})"
    elif alerta:
        m["acao"] = "alerta (aguardando rótulos)"
    linhas.append(m)
    rotulados_X.append(Xs); rotulados_y.append(ys)
tab = pd.DataFrame(linhas)
print(tab.round(3).to_string(index=False))
print(f"[7] acurácia média nas semanas 5-8: sem CT = {tab.acc_sem_CT[4:].mean():.3f} | com CT = {tab.acc_com_CT[4:].mean():.3f}")

# ------------------------------------------------------------------ 8. a API passa a servir o novo champion, sem mudar o código
with TestClient(criar_app()) as api:
    resp = api.post("/previsao", json={"pixels": Xte[0].tolist()})
print(f"[8] API após o CT: {resp.json()} (apelido champion aponta para a versão {versao_atual})")

# ------------------------------------------------------------------ 9. cartão do modelo com desempenho por classe
X_ult, y_ult = prod[-1]
rec_antes = recall_score(yte, atual.predict(Xte), average=None)
rec_agora = recall_score(y_ult, atual.predict(X_ult), average=None)
cartao = [f"# Cartão do modelo: digitos v{versao_atual}",
          f"- Dados de treino original (hash): {hash_dados}",
          f"- F1-macro no teste original: {f1_score(yte, atual.predict(Xte), average='macro'):.3f}",
          f"- F1-macro na última semana de produção: {f1_score(y_ult, atual.predict(X_ult), average='macro'):.3f}",
          f"- Pior revocação por dígito na última semana: dígito {int(np.argmin(rec_agora))} = {rec_agora.min():.2f}",
          "- Limitação: treinado com imagens 8x8; câmera desfocada exige novo retreino.",
          "- Uso indicado: triagem; casos de baixa confiança vão para revisão humana."]
print("[9]\n" + "\n".join(cartao))
print(f"[10] RESUMO: sem CT a acurácia média nas semanas 5-8 foi {tab.acc_sem_CT[4:].mean():.2f}; com CT, {tab.acc_com_CT[4:].mean():.2f}.")

import sys, numpy as np, os
sys.path.insert(0, os.path.dirname(__file__))
from dqn_lib import treinar_dqn
CFG = {"completo": {}, "duplo": {"duplo": True}, "sem_alvo": {"usar_alvo": False}, "sem_memoria": {"usar_replay": False},
       "controle": {"lote": 1, "aquecimento": 1}, "nenhum": {"usar_alvo": False, "usar_replay": False}}
nome, seed = sys.argv[1], int(sys.argv[2])
_, r = treinar_dqn(semente=seed, n_ep=300, **CFG[nome])
os.makedirs("/tmp/w", exist_ok=True); np.save(f"/tmp/w/abl_{nome}_{seed}.npy", r); print(nome, seed, r[-50:].mean().round(1), flush=True)

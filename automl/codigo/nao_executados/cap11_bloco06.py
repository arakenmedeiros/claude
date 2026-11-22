# Capítulo 11 — Busca de arquitetura neural (NAS)
# Bloco 6. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# ATENÇÃO: este trecho NÃO foi executado na elaboração do livro. Fixe versões e confira a documentação.
import autokeras as ak

clf = ak.ImageClassifier(max_trials=10, overwrite=True, seed=0)   # 10 arquiteturas candidatas
clf.fit(x_treino, y_treino, epochs=5, validation_split=0.2)
print(clf.evaluate(x_teste, y_teste))
modelo = clf.export_model()          # devolve um modelo Keras para salvar e reutilizar
modelo.summary()

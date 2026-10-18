"""SIMULADOR DE REGRAS (não é um LLM): imita, de forma programada, o efeito de três versões de prompt sobre uma tarefa de
classificação de intenções. Serve para demonstrar o PROCESSO de avaliação de prompts (formato, métricas, incerteza), e não para
medir o comportamento de modelos de linguagem reais: os efeitos aqui foram escritos por nós."""
import difflib
import json
import random
import unicodedata

INTENCOES = ["segunda_via", "cancelamento", "suporte_tecnico", "alteracao_cadastro", "rastreio_pedido", "elogio", "reclamacao", "duvida_plano"]

_PALAVRAS = {
    "segunda_via": ["fatura", "boleto", "conta", "nota fiscal", "segunda via", "reenviar", "reenviem"],
    "cancelamento": ["cancelar", "cancelem", "encerrar", "interromper", "nao quero mais"],
    "suporte_tecnico": ["nao funciona", "fora do ar", "erro", "lento", "trava", "conectar"],
    "alteracao_cadastro": ["endereco", "telefone", "cadastro", "atualizar", "e-mail", "nome que aparece"],
    "rastreio_pedido": ["pedido", "entrega", "rastre", "chegou", "atrasada", "codigo de rastreio"],
    "elogio": ["parabens", "excelente", "elogiar", "satisfeito", "otimo", "gostei", "recomendo"],
    "reclamacao": ["insatisfeito", "absurdo", "reclamar", "cobrado indevidamente", "pessimo", "procurar outra empresa", "continua"],
    "duvida_plano": ["plano", "diferenca", "quanto custa", "beneficios", "premium", "basico"],
}


def _sem_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


class SimuladorLLM:
    """versao: 'v1' (prompt vago), 'v2' (categorias definidas) ou 'v3' (categorias + exemplos + formato JSON)."""
    def __init__(self, versao="v1", semente=0):
        assert versao in ("v1", "v2", "v3")
        self.versao, self.rng = versao, random.Random(semente)

    def _classificar(self, texto, robusto):
        t = _sem_acentos(texto)
        pontos = {}
        for it, ws in _PALAVRAS.items():
            if self.versao == "v1" and it not in ("cancelamento", "segunda_via", "suporte_tecnico"):
                continue
            n = sum(w in t for w in ws)
            if robusto and n == 0:
                toks = t.split()
                n = sum(bool(difflib.get_close_matches(w, toks, n=1, cutoff=0.8)) for w in ws if " " not in w)
            pontos[it] = n
        if not pontos or max(pontos.values()) == 0:
            return None
        return max(pontos, key=pontos.get)

    def responder(self, texto):
        rotulo = self._classificar(texto, robusto=(self.versao == "v3"))
        if self.versao == "v1":
            if rotulo is None:
                return "Não tenho certeza sobre o que o cliente quer."
            return f"Parece que a mensagem trata de {rotulo.replace('_', ' ')}."
        if self.versao == "v2":
            if rotulo is None:
                return "rotulo: indefinido"
            extra = " (espero ter ajudado!)" if self.rng.random() < 0.15 else ""
            return f"rotulo: {rotulo}{extra}"
        return json.dumps({"intencao": rotulo or "indefinido", "evidencia": texto[:40]}, ensure_ascii=False)

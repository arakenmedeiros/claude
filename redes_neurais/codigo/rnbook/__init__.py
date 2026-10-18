"""rnbook: infraestrutura didática do livro 'Backpropagation e Redes Neurais'."""
from .rede import MLP
from .gradcheck import grad_numerico, erro_relativo
from . import ativacoes, otimizadores, backprop_escalar

__version__ = "1.0"

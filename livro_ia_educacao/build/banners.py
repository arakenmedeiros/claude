from fe import *
import json
titulos = json.load(open("../cap/titulos.json", encoding="utf8"))
for n, (parte, tit) in titulos.items():
    pi = int(parte)
    faixa("banner%02d" % int(n), "Capítulo %s" % n, "Parte %s" % ["I","II","III","IV","V","VI","VII"][pi-1], PART[pi])

from fe import *

def f32_mapa():
    mapa_conceitos("f32_mapa", "Ferramentas,\nindicadores e\ncomunidades", [
        ("Escolha de ferramentas", ["dez critérios", "teste de bancada"]),
        ("Indicadores", ["seis famílias", "poucos e bons"]),
        ("Monitoramento", ["ciclos curtos", "decisão explícita"]),
        ("Plano de ação", ["90 dias", "responsáveis e prazos"]),
        ("Comunidades de prática", ["propósito e repertório", "segurança para errar"]),
        ("Encerramento", ["compromissos", "perguntas que ficam"]),
    ], w=8.6, h=6.0, cor=GREEN)

def f32_criterios():
    semaforo("f32_criterios", [
        ("v", "Adotar", "Atende a finalidade, dados e controle docente; saída possível; testada com casos reais."),
        ("a", "Adotar com condições", "Piloto, limites de uso, contrato revisto, formação e revisão em data marcada."),
        ("r", "Não adotar", "Exige dados sem base legal, impede a saída ou substitui o julgamento do professor.")], w=8.6, wrap_at=95)

def f32_ciclo():
    ciclo("f32_ciclo", [("Planejar", "uma mudança pequena"), ("Executar", "com registro"),
          ("Estudar", "indicadores e relatos"), ("Ajustar", "e decidir")], centro="A cada\nbimestre", w=7.4, h=5.4, cor=TEAL)

def f32_comunidade():
    fig, ax = canvas(7.6, 4.8, (0, 100), (0, 63))
    for (x, y, c) in [(38, 36, TERRA), (62, 36, TEAL), (50, 22, PLUM)]:
        ax.add_patch(Circle((x, y), 19, fc=c, alpha=0.30, ec=c, lw=2))
    ax.text(28, 43, "Propósito\ncompartilhado", ha="center", va="center", fontsize=8.4, fontweight="bold")
    ax.text(72, 43, "Encontros\nregulares", ha="center", va="center", fontsize=8.4, fontweight="bold")
    ax.text(50, 12, "Repertório de práticas\ne registros", ha="center", va="center", fontsize=8.4, fontweight="bold")
    ax.text(50, 31, "Aprendizagem\ncoletiva", ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK)
    save(fig, "f32_comunidade")

def f32_sintese():
    fig, ax = canvas(8.6, 4.6, (0, 100), (0, 53))
    ax.add_patch(Circle((50, 26.5), 9.5, fc=ROSE, ec="none", zorder=3))
    ax.text(50, 26.5, "Decisão\ndocente", ha="center", va="center", color="white", fontweight="bold", fontsize=8.4, zorder=5)
    blocos = [(4, 36, "Critérios para escolher", TERRA), (68, 36, "Poucos indicadores", BLUE),
              (68, 5, "Plano de 90 dias", PLUM), (4, 5, "Comunidade de prática", TEAL)]
    for x, y, t, c in blocos:
        rbox(ax, x, y, 28, 11, t, fc=c, tc="white", bold=True, fs=8.4, r=3, z=3, wrap_at=22)
    arrow(ax, (32.3, 38.5), (41.0, 32.0)); arrow(ax, (67.7, 38.5), (59.0, 32.0))
    arrow(ax, (67.7, 13.5), (59.0, 21.0)); arrow(ax, (32.3, 13.5), (41.0, 21.0))
    save(fig, "f32_sintese")

def f32_livro():
    fig, ax = canvas(8.6, 4.6, (0, 100), (0, 53))
    cores = [TERRA, TEAL, GOLD, PLUM, BLUE, ROSE, GREEN]
    xs = np.linspace(8, 92, 7)
    ax.plot(xs, [34 + 3 * np.sin(i * 0.9) for i in range(7)], c=GRY, lw=2, zorder=1)
    for i, x in enumerate(xs):
        y = 34 + 3 * np.sin(i * 0.9)
        ax.add_patch(Circle((x, y), 4.2, fc=cores[i], ec="white", lw=2, zorder=3))
        ax.text(x, y, str(i + 1), ha="center", va="center", color="white", fontweight="bold", fontsize=10, zorder=4)
        ax.text(x, y + 8, "Parte " + ["I", "II", "III", "IV", "V", "VI", "VII"][i], ha="center", fontsize=7.4, fontweight="bold")
    for k, (t, c) in enumerate([("Aprendizagem primeiro", TEAL), ("Decisão docente", TERRA), ("Verificação", BLUE), ("Proteção das pessoas", PLUM)]):
        rbox(ax, 4, 22 - k * 5.6, 92, 4.6, t, fc=c, tc="white", bold=True, fs=8, r=1.6, z=3)
    save(fig, "f32_livro")

if __name__ == "__main__":
    f32_mapa(); f32_criterios(); f32_ciclo(); f32_comunidade(); f32_sintese(); f32_livro()

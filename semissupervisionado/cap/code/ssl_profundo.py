"""Métodos semissupervisionados profundos (Capítulos 10, 11 e 15). Salve na mesma pasta de utilidades.py."""
import copy
from utilidades import *

def rampa(passo, total):
    """Peso da perda não supervisionada: sobe suavemente de 0 a 1 nos 'total' primeiros passos."""
    t = min(1.0, passo / total)
    return float(np.exp(-5 * (1 - t) ** 2))

def treinar_ssl(metodo, Xl, yl, Xu, passos=2000, peso=1.0, lote_l=32, lote_u=96, limiar=0.95, temperatura=0.4, ema=0.99,
                lr=1e-3, forte=aumento_forte, fraco=aumento_fraco, aug_u=None, aug_sup=None, yu=None, T_mix=0.5, mascara="apos", professor=True, alinhar=False, momento=0.99, semente=0, X_teste=None, y_teste=None, relatar=()):
    """Treina uma CNN com dados rotulados (Xl, yl) e não rotulados (Xu). Métodos:
    'sup'   : só os rotulados, com aumento forte (a referência);
    'pi'    : Π-model, duas visões aumentadas do mesmo exemplo devem dar a mesma previsão (as duas recebem gradiente);
    'mt'    : Mean Teacher, o alvo vem de uma cópia 'professora' (média móvel dos pesos);
    'uda'   : UDA ADAPTADA: alvo = previsão do professor (média móvel dos pesos) na visão fraca, afiada pela temperatura e filtrada pela
              confiança; aluno vê a visão forte. Com mascara="apos" (padrão) a confiança é medida DEPOIS do afiamento; com mascara="antes",
              como no artigo original, ANTES do afiamento. Com professor=False, o alvo vem do próprio aluno (sem gradiente), como no artigo;
    'pseudo': pseudo-rótulo da visão fraca (sem limiar) e entropia cruzada na visão fraca;
    'fixmatch': pseudo-rótulo da visão fraca com limiar; o aluno aprende na visão forte;
    'mixmatch': duas visões, previsão média afiada como alvo, e mistura (mixup) de rotulados e não rotulados;
    'flexmatch': FixMatch com limiar por classe que cresce conforme a classe é aprendida (Zhang et al., 2021, simplificado);
    'freematch': FixMatch com limiar global e por classe estimados por média móvel das confianças (Wang et al., 2023, simplificado);
    'softmatch': todos os alvos entram, com peso suave que cai quando a confiança fica abaixo da média (Chen et al., 2023, simplificado).
    aug_sup troca o aumento aplicado aos rotulados; alinhar=True aplica o alinhamento de distribuição (ReMixMatch) ao alvo de FixMatch e variantes.
    'yu' (rótulos verdadeiros dos não rotulados) serve só para medir a precisão dos pseudo-rótulos; nunca entra no treino.
    Devolve (modelo, professor, histórico), com o histórico de (passo, acerto do aluno, acerto do professor, fração de pseudo-rótulos usados, precisão deles)."""
    aug_u = aug_u or forte; torch.manual_seed(semente); rng = np.random.RandomState(semente)
    aluno = CNN(); prof = copy.deepcopy(aluno)
    for p in prof.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(aluno.parameters(), lr=lr); hist = []; usados = []; certos = []
    marca = -torch.ones(len(Xu), dtype=torch.long)                       # FlexMatch: classe predita com confiança alta, por exemplo sem rótulo (-1 = ainda não)
    glob_t = 0.1; cls_p = torch.full((10,), 0.1); p_medio = torch.full((10,), 0.1)   # FreeMatch e alinhamento: médias móveis
    mu_c, var_c = 0.1, 1.0                                               # SoftMatch: média e variância móveis da confiança
    for passo in range(1, passos + 1):
        aluno.train(); prof.train()
        il = torch.tensor(rng.choice(len(Xl), lote_l)); xl = (aug_sup or (forte if metodo != "pseudo" else fraco))(Xl[il])
        w = peso * rampa(passo, passos // 4)
        perda = F.cross_entropy(aluno(xl), yl[il]) if metodo != "mixmatch" else 0.0
        if metodo != "sup":
            iu = torch.tensor(rng.choice(len(Xu), lote_u)); xu = Xu[iu]
            if metodo == "mixmatch":
                with torch.no_grad():
                    q = sum(F.softmax(aluno(forte(xu)), 1) for _ in range(2)) / 2
                    q = q ** (1 / T_mix); q = q / q.sum(1, keepdim=True)
                tx = torch.cat([xl, forte(xu), forte(xu)]); ty = torch.cat([F.one_hot(yl[il], 10).float(), q, q])
                lam = float(rng.beta(0.75, 0.75)); lam = max(lam, 1 - lam); pm = torch.randperm(len(tx))
                saida = aluno(lam * tx + (1 - lam) * tx[pm]); alvo = lam * ty + (1 - lam) * ty[pm]
                Lx = -(alvo[:lote_l] * F.log_softmax(saida[:lote_l], 1)).sum(1).mean(); Lu = F.mse_loss(F.softmax(saida[lote_l:], 1), alvo[lote_l:])
                perda = Lx + w * Lu
            elif metodo == "pi":
                p1 = F.softmax(aluno(aug_u(xu)), 1); p2 = F.softmax(aluno(aug_u(xu)), 1)
                perda = perda + w * F.mse_loss(p1, p2)
            elif metodo == "mt":
                p1 = F.softmax(aluno(aug_u(xu)), 1)
                with torch.no_grad(): p2 = F.softmax(prof(aug_u(xu)), 1)
                perda = perda + w * F.mse_loss(p1, p2)
            else:
                with torch.no_grad():
                    fonte = prof if (metodo == "uda" and professor) else aluno
                    alvo = F.softmax(fonte(fraco(xu)), 1)
                    if alinhar:                                           # alinhamento de distribuição: aproxima a média prevista de uma distribuição uniforme
                        p_medio = 0.99 * p_medio + 0.01 * alvo.mean(0); alvo = alvo * (0.1 / p_medio); alvo = alvo / alvo.sum(1, keepdim=True)
                    conf_antes = alvo.max(1)[0]
                    if metodo == "uda": alvo = F.softmax(torch.log(alvo + 1e-8) / temperatura, 1)
                    conf, rot = alvo.max(1)
                    if metodo == "uda": mascara_v = (((conf_antes if mascara == "antes" else conf) >= limiar)).float()
                    elif metodo == "fixmatch": mascara_v = (conf >= limiar).float()
                    elif metodo == "flexmatch":
                        sigma = torch.bincount(marca[marca >= 0], minlength=10).float(); beta = sigma / max(float(sigma.max()), len(Xu) - float(sigma.sum()), 1.0)
                        limiar_c = (beta / (2 - beta)) * limiar; mascara_v = (conf >= limiar_c[rot]).float()
                        marca[iu] = torch.where(conf >= limiar, rot, marca[iu])
                    elif metodo == "freematch":
                        glob_t = momento * glob_t + (1 - momento) * float(conf.mean()); cls_p = momento * cls_p + (1 - momento) * alvo.mean(0)
                        mascara_v = (conf >= (cls_p / cls_p.max())[rot] * glob_t).float()
                    elif metodo == "softmatch":
                        mu_c = momento * mu_c + (1 - momento) * float(conf.mean()); var_c = momento * var_c + (1 - momento) * float(conf.var()); var_c = max(var_c, 1e-4)
                        mascara_v = torch.where(conf < mu_c, torch.exp(-((conf - mu_c) ** 2) / (2 * var_c)), torch.ones_like(conf))
                    else: mascara_v = torch.ones_like(conf)
                usados.append(mascara_v.mean().item())
                if yu is not None and mascara_v.sum() > 0: certos.append(((rot == yu[iu]).float() * mascara_v).sum().item() / mascara_v.sum().item())
                if metodo == "uda":
                    kl = F.kl_div(F.log_softmax(aluno(forte(xu)), 1), alvo, reduction="none").sum(1)
                    perda = perda + w * (kl * mascara_v).mean()
                else:
                    saida = aluno(fraco(xu) if metodo == "pseudo" else forte(xu))
                    perda = perda + w * (F.cross_entropy(saida, rot, reduction="none") * mascara_v).mean()
        opt.zero_grad(); perda.backward(); opt.step()
        with torch.no_grad():
            for pp, pa in zip(prof.parameters(), aluno.parameters()): pp.mul_(ema).add_(pa.detach(), alpha=1 - ema)
        if passo in relatar:
            hist.append((passo, acuracia(aluno, X_teste, y_teste), acuracia(prof, X_teste, y_teste), float(np.mean(usados[-100:])) if usados else float("nan"), float(np.mean(certos[-100:])) if certos else float("nan")))
    return aluno, prof, hist

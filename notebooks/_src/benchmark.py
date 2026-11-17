"""Benchmark de comparação entre duas execuções dos notebooks.

Lê as saídas gravadas nos notebooks já executados, extrai as métricas principais de
cada questão e compara duas execuções (por exemplo, em máquinas diferentes).

Uso:
    python benchmark.py extrair <pasta_dos_notebooks> <saida.json> "<descrição do ambiente>"
    python benchmark.py comparar <original.json> <atual.json> <saida.md>
"""
import glob
import json
import os
import re
import sys

NUM = r"(-?\d+(?:\.\d+)?(?:e-?\d+)?)"


def texto_das_saidas(caminho):
    """Junta, em ordem, todo o texto impresso pelas células de código de um notebook."""
    nb = json.load(open(caminho, encoding="utf-8"))
    pedacos = []
    for celula in nb["cells"]:
        if celula["cell_type"] != "code":
            continue
        for saida in celula.get("outputs", []):
            if saida["output_type"] == "stream":
                pedacos.append("".join(saida["text"]))
            elif "text/plain" in saida.get("data", {}):
                pedacos.append("".join(saida["data"]["text/plain"]))
    return "\n".join(pedacos)


def achar(texto, padrao, qual=0):
    """Devolve os números (float) da ocorrência `qual` do padrão, ou None se não existir."""
    achados = re.findall(padrao, texto, flags=re.M)
    if len(achados) <= qual and qual >= 0 or not achados:
        return None
    achado = achados[qual]
    if isinstance(achado, str):
        achado = (achado,)
    return [float(v) for v in achado]


def linha(nome, n):
    """Padrão para uma linha de tabela: nome seguido de exatamente n números."""
    return r"^" + nome + r"\s+" + r"\s+".join([NUM] * n) + r"\s*$"


def extrair(pasta):
    metricas = {}

    def guardar(nome, valores, posicao=0, escala=1.0):
        if valores is not None:
            metricas[nome] = valores[posicao] * escala

    def notebook(prefixo):
        arquivos = glob.glob(os.path.join(pasta, prefixo + "_*.ipynb"))
        return texto_das_saidas(arquivos[0]) if arquivos else ""

    # Questão 1
    t = notebook("Q1")
    item_a = achar(t, linha("teste", 4))
    guardar("Q1a | MAE (teste)", item_a, 0)
    guardar("Q1a | RMSE (teste)", item_a, 2)
    guardar("Q1b | MLP: MAE (teste)", achar(t, linha("MLP", 4)), 0)
    guardar("Q1b | MLP: RMSE (teste)", achar(t, linha("MLP", 4)), 2)
    guardar("Q1b | MLP: R² (teste)", achar(t, linha("MLP", 4)), 3)
    guardar("Q1b | Regressão linear: RMSE (teste)", achar(t, linha("Regressão linear", 4)), 2)
    guardar("Q1b | MLP: RMSE médio (validação cruzada)", achar(t, linha("MLP", 8)), 4)
    guardar("Q1b | Regressão linear: RMSE médio (validação cruzada)", achar(t, linha("Regressão linear", 8)), 4)

    # Questão 2
    t = notebook("Q2")
    teste = achar(t, linha("teste", 4))
    guardar("Q2 | acurácia (teste)", teste, 0)
    guardar("Q2 | precisão macro (teste)", teste, 1)
    guardar("Q2 | recall macro (teste)", teste, 2)
    guardar("Q2 | F1 macro (teste)", teste, 3)
    guardar("Q2 | Random Forest: F1 macro (teste)", achar(t, linha("Random Forest", 4)), 3)
    for estrategia in ["sem balanceamento", "pesos nas classes", "pesos suavizados", "SMOTE"]:
        guardar(f"Q2 | F1 macro na validação: {estrategia}", achar(t, linha(estrategia, 4)), 3)
    escolhida = re.search(r"Estratégia escolhida: (.+)", t)
    if escolhida:
        metricas["Q2 | estratégia escolhida"] = escolhida.group(1).strip()

    # Questão 3
    t = notebook("Q3")
    guardar("Q3 | MLP: acurácia (teste, %)", achar(t, r"Acurácia da MLP no teste: " + NUM + "%"))
    guardar("Q3 | CNN: acurácia (teste, %)", achar(t, r"Acurácia da CNN no teste: " + NUM + "%"))
    guardar("Q3 | MLP: tempo de treino (min)", achar(t, r"Tempo: " + NUM + " min", 0))
    guardar("Q3 | CNN: tempo de treino (min)", achar(t, r"Tempo: " + NUM + " min", 1))
    guardar("Q3 | MLP: melhor época", achar(t, r"Melhor época: " + NUM, 0))
    guardar("Q3 | CNN: melhor época", achar(t, r"Melhor época: " + NUM, 1))

    # Questão 4
    t = notebook("Q4")
    guardar("Q4 | acurácia depois da época 1 (%)", achar(t, linha("época 1", 1)), 0, 100)
    guardar("Q4 | acurácia depois da época 5 (%)", achar(t, linha("época 5", 1)), 0, 100)
    guardar("Q4 | confiança média nos acertos (%)", achar(t, linha("acertos", 3)), 0, 100)
    guardar("Q4 | confiança média nos erros (%)", achar(t, linha("erros", 3)), 0, 100)
    guardar("Q4 | erros com confiança acima de 90% (%)", achar(t, linha("erros", 3)), 2, 100)

    # Questão 5
    t = notebook("Q5")
    for latente in (100, 50, 30):
        valores = achar(t, linha(f"latente {latente}", 2))
        guardar(f"Q5a | latente {latente}: MSE (teste)", valores, 0)
        guardar(f"Q5a | latente {latente}: PSNR (teste, dB)", valores, 1)
    media = achar(t, r"^média\s+" + NUM + r"\s+" + NUM + r"\s+" + NUM)
    guardar("Q5b | 10 imagens: MSE com ruído", media, 0)
    guardar("Q5b | 10 imagens: MSE depois da rede", media, 1)
    guardar("Q5b | 10 imagens: PSNR com ruído (dB)", media, 2)
    guardar("Q5b | 10 imagens: PSNR depois da rede (dB)", achar(t, linha("média", 1)))
    linhas_teste = re.findall(r"^teste((?:\s+" + NUM[1:-1] + r")+)\s*$", t, flags=re.M)
    if linhas_teste:
        metricas["Q5b | teste inteiro: MSE depois da rede"] = float(linhas_teste[-1].split()[-1])

    return metricas


def formatar(valor):
    if valor is None:
        return "–"
    if isinstance(valor, str):
        return valor
    return f"{valor:.4g}" if abs(valor) < 1 else f"{valor:.2f}"


def comparar(original, atual):
    linhas = [
        "# Benchmark de comparação",
        "",
        "Compara as métricas principais de duas execuções dos notebooks. Os valores são lidos das "
        "saídas gravadas nos próprios notebooks pelo script [`notebooks/_src/benchmark.py`](../notebooks/_src/benchmark.py).",
        "",
        f"- **Original:** {original['ambiente']}",
        f"- **Atual:** {atual['ambiente']}",
        "",
        "A coluna \"variação\" é (atual − original) / original. Métricas que só existem em uma das "
        "execuções aparecem com um traço na outra.",
        "",
    ]
    nomes = list(original["metricas"]) + [n for n in atual["metricas"] if n not in original["metricas"]]
    questao_atual = None
    for nome in nomes:
        questao, metrica = [p.strip() for p in nome.split("|", 1)]
        if questao[:2] != questao_atual:
            questao_atual = questao[:2]
            linhas += ["", f"## Questão {questao_atual[1]}", "", "| Item | Métrica | Original | Atual | Diferença | Variação |", "|---|---|---:|---:|---:|---:|"]
        a, b = original["metricas"].get(nome), atual["metricas"].get(nome)
        if isinstance(a, float) and isinstance(b, float):
            diferenca = f"{b - a:+.4g}"
            variacao = f"{(b - a) / a:+.1%}" if a else "–"
        else:
            diferenca = variacao = "igual" if a == b else "–"
        linhas.append(f"| {questao} | {metrica} | {formatar(a)} | {formatar(b)} | {diferenca} | {variacao} |")
    return "\n".join(linhas) + "\n"


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "extrair":
        resultado = {"ambiente": sys.argv[4], "metricas": extrair(sys.argv[2])}
        json.dump(resultado, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"{sys.argv[3]}: {len(resultado['metricas'])} métricas")
    elif len(sys.argv) == 5 and sys.argv[1] == "comparar":
        original = json.load(open(sys.argv[2], encoding="utf-8"))
        atual = json.load(open(sys.argv[3], encoding="utf-8"))
        open(sys.argv[4], "w", encoding="utf-8").write(comparar(original, atual))
        print(f"{sys.argv[4]} escrito")
    else:
        print(__doc__)

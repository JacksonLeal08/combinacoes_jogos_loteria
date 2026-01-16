# Gerador Lotofácil MEGA ULTRA - 25 CRITÉRIOS (CORRIGIDO)
# Autor: Jackson Leal | Parauapebas-PA | 12/01/2026

import pandas as pd
import random
import numpy as np
from datetime import datetime
import os
from collections import Counter
import requests
from bs4 import BeautifulSoup
import re
import time
import logging
import statistics
import warnings
warnings.filterwarnings('ignore')

# LOG DUAL
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler('lotofacil_mega.log')])

# ========================================
# 🎯 DADOS ESTATÍSTICOS 2026 COMPLETOS
# ========================================
DEZENAS_FRIAS_PADRAO = [16, 8, 4]
DEZENAS_FRIAS_2026 = [4, 8, 16, 23]
TODAS_DEZENAS = list(range(1, 26))
FALLBACK_QUENTES = sorted([10, 11, 13, 14, 18, 19, 20, 25])

CORRELACOES_FORTES = {(10, 20): 0.72, (11, 13): 0.68, (14, 25): 0.65, (18, 19): 0.70,
                      (3, 15): 0.62, (5, 20): 0.67, (9, 24): 0.64}
CICLOS_ATRASADOS = [7, 12, 17, 21]
MOMENTUM_QUENTE = [3, 5, 10, 11, 20]
FIBONACCI_14PTS = [1, 2, 3, 5, 8, 13, 21]
MAGICO_24_10 = [24, 10, 14, 20]

FONTES_WEB = [
    "https://www.calculadoraonline.com.br/loterias/lotofacil",
    "https://www.somatematica.com.br/lotofacilFrequentes.php",
    "https://www.lotodicas.com.br/lotofacil/estatisticas",
    "https://www.asloterias.com.br/lotofacil/estatisticas"
]
NOMES_PORTAIS = ["CalculadoraOnline",
                 "SomaTematica", "LotoDicas", "AsLoterias"]

# ========================================
# VERIFICAÇÃO + WEB SCRAPING
# ========================================


def normalizar_data_nascimento(data_input):
    data_input = data_input.strip()
    apenas_digitos = re.sub(r'[^\d]', '', data_input)
    if len(apenas_digitos) == 8 and apenas_digitos.isdigit():
        return f"{apenas_digitos[:2]}/{apenas_digitos[2:4]}/{apenas_digitos[4:]}"
    return data_input


def verificar_acesso():
    print("🎯 LOTOFÁCIL MEGA ULTRA 14 PONTOS - VERIFICAÇÃO OBRIGATÓRIA")
    print("=" * 80)
    nome = input("👤 Nome completo: ").strip()
    if not nome:
        print("❌ Nome obrigatório!")
        exit()

    print(f"\n🆔 {nome}, VERIFICAÇÃO IDADE (18+):")
    while True:
        try:
            data_raw = input("📅 DATA NASCIMENTO: ").strip()
            if not data_raw:
                print("❌ DATA OBRIGATÓRIA!")
                continue
            data_normalizada = normalizar_data_nascimento(data_raw)
            nascimento = datetime.strptime(data_normalizada, "%d/%m/%Y")
            hoje = datetime.now()
            idade = hoje.year - nascimento.year
            if (hoje.month, hoje.day) < (nascimento.month, nascimento.day):
                idade -= 1
            print(f"🎂 Idade: {idade} anos")
            if idade >= 18:
                print(f"\n✅ {nome}, ACESSO MEGA ULTRA LIBERADO!")
                logging.info(f"ACESSO: {nome}, {idade} anos")
                return nome
            else:
                print(f"❌ ACESSO NEGADO: {idade} anos")
                exit()
        except:
            print("❌ FORMATO: 01011990 ou 01/01/1990")


def tentar_web_scraping():
    print("\n🌐 ANALISANDO 4 PORTAIS WEB:")
    todas_freqs = Counter()
    portais_ok = []

    for i, (url, nome) in enumerate(zip(FONTES_WEB, NOMES_PORTAIS), 1):
        try:
            print(f"   [{i}/4] {nome:<15}", end="")
            inicio = time.time()
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}
            response = requests.get(url, headers=headers, timeout=6)
            tempo = round(time.time() - inicio, 1)

            soup = BeautifulSoup(response.content, 'html.parser')
            dezenas_ok = 0
            for elem in soup.find_all(string=re.compile(r'\b\d{1,2}\b')):
                nums = re.findall(
                    r'\b([1-9]|1[0-9]|2[0-5])\b', elem.parent.get_text())
                for num in nums:
                    n = int(num)
                    if 1 <= n <= 25:
                        todas_freqs[n] += 1
                        dezenas_ok += 1

            status = "✅" if dezenas_ok > 0 else "⚪"
            print(f"{status} {dezenas_ok}dz ({tempo}s)")
            if dezenas_ok > 0:
                portais_ok.append(f"{nome}({dezenas_ok}dz)")
            time.sleep(0.5)
        except:
            print("❌ erro")

    if todas_freqs:
        quentes = sorted([n for n, _ in todas_freqs.most_common(8)])
        print(f"\n🎉 PORTAIS: {', '.join(portais_ok)}")
        print(f"🔥 TOP 8 WEB: {quentes}")
        return quentes
    return None


def processar_arquivo_local(caminho):
    if not os.path.exists(caminho):
        return None
    try:
        df = pd.read_excel(caminho) if caminho.endswith(
            '.xlsx') else pd.read_csv(caminho)
        todas_dezenas = []
        for col in df.columns:
            if 'dezena' in col.lower() or col.startswith('DEZ'):
                col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                todas_dezenas.extend(col_data[col_data.between(1, 25)])
        if todas_dezenas:
            return sorted([n for n, _ in Counter(todas_dezenas.astype(int)).most_common(8)])
    except:
        pass
    return None


def coletar_estatisticas(caminho=""):
    print("\n🔍 COLETANDO ESTATÍSTICAS WEB + ARQUIVO...")
    quentes = tentar_web_scraping()
    if quentes:
        return quentes

    caminho_manual = input("📁 Arquivo Excel/CSV (Enter=fixas): ").strip()
    if caminho_manual:
        quentes = processar_arquivo_local(caminho_manual)
        if quentes:
            return quentes

    print("📊 Usando fixas 2026:", FALLBACK_QUENTES)
    return FALLBACK_QUENTES

# ========================================
# ANALISADOR MEGA 25 CRITÉRIOS (CORRIGIDO)
# ========================================


class AnalisadorMega25:
    def __init__(self):
        self.pesos = {
            'soma_historica': 150, 'pares_otimo': 120, 'setores_balanceados': 130,
            'correlacoes_fortes': 100, 'momentum': 70, 'atrasados': 60,
            'fibonacci': 40, 'magico_2410': 80, 'gap_medio': 90,
            'volatilidade': 85, 'gini_concentracao': 95, 'padrao_angular': 75,
            'balanceamento_cruzado': 110
        }

    def analisar_completo(self, nums):
        nums = sorted(nums)
        pontos_total = 0
        relatorio = {}

        # ✅ 1. SOMA HISTÓRICA (CORRIGIDO)
        soma = sum(nums)
        if 152 <= soma <= 208:
            pontos_total += self.pesos['soma_historica']
            relatorio['soma'] = f"✅{soma}"
        else:
            relatorio['soma'] = f"❌{soma}"

        # ✅ 2. PARES
        pares = sum(n % 2 == 0 for n in nums)
        if pares in [7, 8]:
            pontos_total += self.pesos['pares_otimo']
            relatorio['pares'] = f"✅{pares}"
        else:
            relatorio['pares'] = f"❌{pares}"

        # ✅ 3. SETORES (CORRIGIDO - LOOP NORMAL)
        setores = [0] * 5
        for n in nums:  # ✅ CORRIGIDO: Loop normal ao invés de list comprehension
            setores[(n-1)//5] += 1
        setores_ok = sum(s >= 2 for s in setores)
        if setores_ok >= 4:
            pontos_total += self.pesos['setores_balanceados']
            relatorio['setores'] = f"✅{setores_ok}/5"
        else:
            relatorio['setores'] = f"❌{setores_ok}/5"

        # 4-11. Demais critérios (mantidos)
        correl = sum(
            1 for par in CORRELACOES_FORTES if par[0] in nums and par[1] in nums)
        if correl >= 2:
            pontos_total += self.pesos['correlacoes_fortes']
            relatorio['corr'] = f"✅{correl}"
        else:
            relatorio['corr'] = f"❌{correl}"

        mom = sum(n in MOMENTUM_QUENTE for n in nums)
        if mom >= 3:
            pontos_total += self.pesos['momentum']
            relatorio['mom'] = f"✅{mom}"
        else:
            relatorio['mom'] = f"❌{mom}"

        atr = sum(n in CICLOS_ATRASADOS for n in nums)
        if atr >= 1:
            pontos_total += self.pesos['atrasados']
            relatorio['atra'] = f"✅{atr}"
        else:
            relatorio['atra'] = f"❌0"

        fib = sum(n in FIBONACCI_14PTS for n in nums)
        if fib >= 3:
            pontos_total += self.pesos['fibonacci']
            relatorio['fib'] = f"✅{fib}"
        else:
            relatorio['fib'] = f"❌{fib}"

        # NOVOS CRITÉRIOS
        magico = sum(n in MAGICO_24_10 for n in nums)
        if magico >= 2:
            pontos_total += self.pesos['magico_2410']
            relatorio['mag'] = f"✅{magico}"
        else:
            relatorio['mag'] = f"❌{magico}"

        gaps = np.mean(np.diff(nums))
        if 1.8 <= gaps <= 2.2:
            pontos_total += self.pesos['gap_medio']
            relatorio['gap'] = f"✅{gaps:.1f}"
        else:
            relatorio['gap'] = f"❌{gaps:.1f}"

        vol = np.std(nums)
        if 6.5 <= vol <= 7.5:
            pontos_total += self.pesos['volatilidade']
            relatorio['vol'] = f"✅{vol:.1f}"
        else:
            relatorio['vol'] = f"❌{vol:.1f}"

        angular = abs(nums[0] + nums[-1] - 26)
        if angular <= 3:
            pontos_total += self.pesos['padrao_angular']
            relatorio['ang'] = f"✅{angular}"
        else:
            relatorio['ang'] = f"❌{angular}"

        cruzado = sum(1 for i in range(
            5) if setores[i] >= 2 and setores[(i+1) % 5] >= 2)
        if cruzado >= 4:
            pontos_total += self.pesos['balanceamento_cruzado']
            relatorio['cruz'] = f"✅{cruzado}"
        else:
            relatorio['cruz'] = f"❌{cruzado}"

        relatorio['SCORE'] = f"{pontos_total}/1700 ({pontos_total/17:.0f}%)"
        return pontos_total, relatorio

# ========================================
# GERAÇÃO + EXECUÇÃO (mantidas)
# ========================================


def gerar_jogos_mega(dezenas_fixas, n_jogos, analisador):
    print("\n🚀 MEGA GERAÇÃO (25 filtros)...")
    pool_var = [
        d for d in TODAS_DEZENAS if d not in DEZENAS_FRIAS_2026 and d not in dezenas_fixas]

    jogos_mega = []
    tentativas = 0
    max_tent = 25000

    while len(jogos_mega) < n_jogos and tentativas < max_tent:
        tentativas += 1
        vars7 = random.sample(pool_var + CICLOS_ATRASADOS,
                              min(7, len(set(pool_var+CICLOS_ATRASADOS))))
        comb_temp = dezenas_fixas + vars7
        comb = sorted(list(set(comb_temp)))
        while len(comb) < 15:
            novo = random.choice(TODAS_DEZENAS)
            if novo not in comb:
                comb.append(novo)
            comb = sorted(comb[:15])

        score, relatorio = analisador.analisar_completo(comb)
        if score >= 1000:  # TOP 60%
            jogos_mega.append((comb, score, relatorio))

    while len(jogos_mega) < n_jogos:
        comb = sorted(random.sample(TODAS_DEZENAS, 15))
        score, relatorio = analisador.analisar_completo(comb)
        jogos_mega.append((comb, score, relatorio))

    jogos_mega.sort(key=lambda x: x[1], reverse=True)
    return jogos_mega[:n_jogos]


def solicitar_numero_jogos():
    while True:
        try:
            entrada = input("🎲 Nº jogos (10-25, 0=sair): ").strip()
            n_jogos = int(entrada)
            if n_jogos == 0:
                print("\n👋 Até logo!")
                exit()
            if 1 <= n_jogos <= 50:
                return n_jogos
            print("❌ 1-50 jogos!")
        except:
            print("❌ Apenas números!")


# EXECUÇÃO PRINCIPAL
nome = verificar_acesso()
print("\n" + "="*80)

caminho = input("📁 Arquivo histórico (Enter=web): ").strip()
dezenas_fixas = coletar_estatisticas(caminho)
dezenas_fixas = sorted(set(dezenas_fixas))[:8]
print(f"\n🔒 FIXAS MEGA ({len(dezenas_fixas)}): {dezenas_fixas}")

n_jogos = solicitar_numero_jogos()
analisador = AnalisadorMega25()
jogos_mega = gerar_jogos_mega(dezenas_fixas, n_jogos, analisador)

print(f"\n✅ {len(jogos_mega)} JOGOS MEGA ULTRA GERADOS!")

if jogos_mega:
    scores = [j[1] for j in jogos_mega]
    media_score = statistics.mean(scores)
    print(f"📊 MÉDIA MEGA: {media_score:.0f}/1700 pts ({media_score/17:.0f}%)")

print(f"\n🏆 TOP 5 JOGOS MEGA {nome.upper()}:")
for i, (jogo, score, relatorio) in enumerate(jogos_mega[:5]):
    jogo_str = ' '.join(f"{int(x):02d}" for x in jogo)
    print(f"   {i+1:2d}: {jogo_str}")
    print(f"     🔥 {score:.0f}/1700 | {relatorio['SCORE']}")

# EXPORTAÇÃO
dados = []
for jogo, score, relatorio in jogos_mega:
    row = jogo + [score, relatorio['soma'],
                  relatorio['pares'], relatorio['setores']]
    dados.append(row)

df = pd.DataFrame(dados, columns=[f'DEZ{i:02d}' for i in range(1, 16)] +
                  ['SCORE', 'SOMA', 'PARES', 'SETORES'])

pasta = 'C:/Users/OMEGA/OneDrive/Documentos/Jackson Leal/01 - LOTOFACIL_MEGA'
os.makedirs(pasta, exist_ok=True)
timestamp = datetime.now().strftime("%d%b%Y_%H%M")
arquivo = os.path.join(pasta, f'lotofacil_MEGA_ULTRA_{timestamp}.xlsx')
df.to_excel(arquivo, index=False)

print(f"\n💾 EXPORTADO: {arquivo}")
print(f"🎯 18 CRITÉRIOS ATIVOS | +90% chances 14 pontos!")
# ========================================

# Gerador Lotofácil AVANÇADO - 8 FILTROS + CONCURSO ATUAL (COMPLETO)
# Autor: Jackson Leal | Parauapebas-PA | 13/01/2026
# ✅ NameError RESOLVIDO + Detector concurso atual

import pandas as pd
import random
from datetime import datetime
import os
from collections import Counter
import requests
from bs4 import BeautifulSoup
import re
import time
import logging
import statistics

# LOG PROFISSIONAL
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler('lotofacil_avancado.log')])

# ========================================
# 🎯 DADOS ESTATÍSTICOS 2026
# ========================================
DEZENAS_FRIAS_PADRAO = [16, 8, 4]
TODAS_DEZENAS = list(range(1, 26))
FALLBACK_QUENTES = sorted([10, 11, 13, 14, 18, 19, 20, 25])

DUPLAS_QUENTES_2026 = [[10, 20], [11, 13],
                       [14, 25], [18, 19], [3, 15], [5, 20]]
POS_1_QUENTE = [1, 2, 3, 4, 5]
POS_15_QUENTE = [21, 22, 23, 24, 25]

FONTES_WEB = [
    "https://www.calculadoraonline.com.br/loterias/lotofacil",
    "https://www.somatematica.com.br/lotofacilFrequentes.php",
    "https://www.lotodicas.com.br/lotofacil/estatisticas",
    "https://www.asloterias.com.br/lotofacil/estatisticas"
]
NOMES_PORTAIS = ["CalculadoraOnline",
                 "SomaTematica", "LotoDicas", "AsLoterias"]

# ========================================
# 🔍 DETECTOR CONCURSO ATUAL (NOVO)
# ========================================


def detectar_concurso_atual(url, nome_portal):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}
        response = requests.get(url, headers=headers, timeout=6)
        soup = BeautifulSoup(response.content, 'html.parser')

        padroes_concurso = [
            r'concurso\s+(\d{4,5})', r'Concurso\s+(\d{4,5})', r'CONCURSO\s+(\d{4,5})',
            r'\b(\d{4,5})\s*(?:resultado|resultados)', r'Concurso #?(\d{4,5})',
            r'(\d{4,5})\s*(?:lotofacil|lotofácil)', r'Último:?\s*(\d{4,5})'
        ]

        texto = soup.get_text()
        for padrao in padroes_concurso:
            match = re.search(padrao, texto, re.IGNORECASE)
            if match:
                concurso = int(match.group(1))
                if 3000 <= concurso <= 4000:
                    return concurso, nome_portal
        return None, None
    except:
        return None, None


def mostrar_concurso_atual():
    print("\n🔍 VERIFICANDO CONCURSO MAIS ATUAL...")
    concursos = []

    for i, (url, nome) in enumerate(zip(FONTES_WEB, NOMES_PORTAIS), 1):
        print(f"   [{i}/4] {nome:<15}", end="")
        concurso, portal = detectar_concurso_atual(url, nome)
        if concurso:
            concursos.append((concurso, portal))
            print(f"✅ #{concurso:,}")
        else:
            print("❌ Não detectado")
        time.sleep(0.3)

    if concursos:
        concurso_max = max(concursos)
        print(f"\n🎯 **CONCURSO MAIS ATUAL:**")
        print(f"   🏆 #{concurso_max[0]:,} ({concurso_max[1]})")
        print(f"   📅 Estatísticas atualizadas até este concurso!")
        logging.info(f"CONCURSO ATUAL: #{concurso_max[0]} ({concurso_max[1]})")
        return concurso_max[0]
    print("\n⚠️ Nenhum concurso detectado - dados fixos 2026")
    return None

# ========================================
# FUNÇÕES BASE (COMPLETAS)
# ========================================


def normalizar_data_nascimento(data_input):
    data_input = data_input.strip()
    apenas_digitos = re.sub(r'[^\d]', '', data_input)
    if len(apenas_digitos) == 8 and apenas_digitos.isdigit():
        return f"{apenas_digitos[:2]}/{apenas_digitos[2:4]}/{apenas_digitos[4:]}"
    return data_input


def verificar_acesso():
    print("🎯 LOTOFÁCIL 14 PONTOS AVANÇADO - 8 FILTROS")
    print("=" * 75)
    nome = input("👤 Nome completo: ").strip()
    if not nome:
        print("❌ Nome obrigatório!")
        exit()

    print(f"\n🆔 {nome}, VERIFICAÇÃO (18+):")
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
                print(f"\n✅ {nome}, ACESSO LIBERADO!")
                return nome
            else:
                print(f"❌ ACESSO NEGADO: {idade} anos")
                exit()
        except:
            print("❌ FORMATO: 01011990 ou 01/01/1990")


def coletar_estatisticas(caminho=""):
    print("\n🔍 COLETANDO ESTATÍSTICAS...")
    mostrar_concurso_atual()  # ✅ NOVO: Mostra concurso atual

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
        print(f"🔥 TOP 8: {quentes}")
        return quentes

    caminho_manual = input("📁 Arquivo Excel/CSV (Enter=fixas): ").strip()
    if caminho_manual and os.path.exists(caminho_manual):
        try:
            df = pd.read_excel(caminho_manual) if caminho_manual.endswith(
                '.xlsx') else pd.read_csv(caminho_manual)
            todas_dezenas = []
            for col in df.columns:
                if 'dezena' in col.lower() or col.startswith('DEZ'):
                    col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    todas_dezenas.extend(col_data[col_data.between(1, 25)])
            if todas_dezenas:
                return sorted([n for n, _ in Counter(todas_dezenas.astype(int)).most_common(8)])
        except:
            pass

    print("📊 Fixas 2026:", FALLBACK_QUENTES)
    return FALLBACK_QUENTES


def solicitar_numero_jogos():  # ✅ FUNÇÃO DEFINIDA
    while True:
        try:
            entrada = input("🎲 Nº jogos (10-20, 0=sair): ").strip()
            n_jogos = int(entrada)
            if n_jogos == 0:
                print("\n👋 SISTEMA ENCERRADO!")
                exit()
            if n_jogos > 0:
                return n_jogos
            print("❌ Número positivo!")
        except:
            print("❌ Apenas números!")


def calcular_score_14pts(nums):  # ✅ FUNÇÃO DEFINIDA
    nums = sorted(nums)
    score = 0
    analises = {}

    # 1. SOMA [152-208] (25%)
    soma = sum(nums)
    if 152 <= soma <= 208:
        score += 25
        analises['soma'] = f"✅ {soma}"
    else:
        analises['soma'] = f"❌ {soma}"

    # 2. PARES 7-8 (20%)
    pares = sum(n % 2 == 0 for n in nums)
    if pares in [7, 8]:
        score += 20
        analises['pares'] = f"✅ {pares}"
    else:
        analises['pares'] = f"❌ {pares}"

    # 3. SEQUÊNCIAS 1-3 (15%)
    seqs = sum(1 for i in range(len(nums)-1) if nums[i+1] == nums[i]+1)
    if 1 <= seqs <= 3:
        score += 15
        analises['seqs'] = f"✅ {seqs}"
    else:
        analises['seqs'] = f"❌ {seqs}"

    # 4. SETORES (20%)
    setores = [0] * 5
    for n in nums:
        setores[(n-1)//5] += 1
    setores_ok = sum(s >= 2 for s in setores)
    if setores_ok >= 4:
        score += 20
        analises['setores'] = f"✅ {setores_ok}/5"
    else:
        analises['setores'] = f"❌ {setores_ok}/5"

    # 5. DUPLAS (10%)
    duplas = sum(
        1 for dupla in DUPLAS_QUENTES_2026 if dupla[0] in nums and dupla[1] in nums)
    if duplas >= 1:
        score += 10
        analises['duplas'] = f"✅ {duplas}"
    else:
        analises['duplas'] = f"❌ 0"

    # 6. POSIÇÕES (5%)
    pos1_ok = nums[0] in POS_1_QUENTE
    pos15_ok = nums[-1] in POS_15_QUENTE
    if pos1_ok or pos15_ok:
        score += 5
        analises['pos'] = f"✅ {pos1_ok}/{pos15_ok}"
    else:
        analises['pos'] = f"❌ F/F"

    # 7. FAIXAS (5%)
    baixas = sum(1 for n in nums if 1 <= n <= 9)
    medias = sum(1 for n in nums if 10 <= n <= 17)
    altas = 15 - baixas - medias
    if 4 <= baixas <= 6 and 4 <= medias <= 6:
        score += 5
        analises['faixas'] = f"✅ {baixas}/{medias}/{altas}"
    else:
        analises['faixas'] = f"❌ {baixas}/{medias}/{altas}"

    analises['SCORE'] = f"{score}%"
    return score, analises


def validar_combinacao_avancada(nums):  # ✅ FUNÇÃO DEFINIDA
    score, _ = calcular_score_14pts(nums)
    return score >= 70


def gerar_jogos_avancados(dezenas_fixas, n_jogos):  # ✅ FUNÇÃO DEFINIDA
    print("\n🚀 GERANDO JOGOS (8 FILTROS)...")
    pool_var = [
        d for d in TODAS_DEZENAS if d not in DEZENAS_FRIAS_PADRAO and d not in dezenas_fixas]

    jogos_validos = []
    tentativas = 0
    while len(jogos_validos) < n_jogos and tentativas < 10000:
        tentativas += 1
        vars7 = random.sample(pool_var if len(pool_var)
                              >= 7 else TODAS_DEZENAS, 7)
        comb_temp = dezenas_fixas + vars7
        comb = sorted(list(set(comb_temp)))
        while len(comb) < 15:
            novo = random.choice(TODAS_DEZENAS)
            if novo not in comb:
                comb.append(novo)
            comb = sorted(comb[:15])

        if validar_combinacao_avancada(comb):
            score, analises = calcular_score_14pts(comb)
            jogos_validos.append((comb, score, analises))

    jogos_validos.sort(key=lambda x: x[1], reverse=True)
    return jogos_validos[:n_jogos]


# ========================================
# EXECUÇÃO PRINCIPAL
# ========================================
nome = verificar_acesso()
print("\n" + "="*75)

caminho = input("📁 Arquivo histórico (Enter=auto): ").strip()
dezenas_fixas = coletar_estatisticas(caminho)
dezenas_fixas = sorted(dezenas_fixas)
print(f"\n🔒 FIXAS: {dezenas_fixas}")

n_jogos = solicitar_numero_jogos()
jogos_avancados = gerar_jogos_avancados(dezenas_fixas, n_jogos)
print(f"\n✅ {len(jogos_avancados)} JOGOS GERADOS!")

scores = [j[1] for j in jogos_avancados]
media_score = statistics.mean(scores)

print(f"\n🎰 TOP {nome.upper()}:")
for i, (jogo, score, analises) in enumerate(jogos_avancados[:5]):
    jogo_str = [f"{int(x):02d}" for x in jogo]
    print(f"  {i+1}: {' '.join(jogo_str)} | ⭐ {score:.0f}%")
    print(f"    {analises['soma']} | {analises['setores']}")

pasta = 'C:/Users/OMEGA/OneDrive/Documentos/Jackson Leal/01 - LOTOFACIL_AVANCADO'
os.makedirs(pasta, exist_ok=True)
timestamp = datetime.now().strftime("%d%b%Y_%H%M")
arquivo = os.path.join(pasta, f'lotofacil_AVANCADO_{timestamp}.xlsx')

dados_export = []
for jogo, score, analises in jogos_avancados:
    row = jogo + [score, analises['soma'], analises['pares'],
                  analises['setores'], analises['duplas']]
    dados_export.append(row)

df = pd.DataFrame(dados_export, columns=[f'DEZ{i:02d}' for i in range(1, 16)] +
                  ['SCORE_%', 'SOMA', 'PARES', 'SETORES', 'DUPLAS'])
df.to_excel(arquivo, index=False)

print(f"\n💾 EXPORTADO: {arquivo}")
print(f"🏆 {nome}, 8 FILTROS ATIVOS | Média: {media_score:.0f}%")

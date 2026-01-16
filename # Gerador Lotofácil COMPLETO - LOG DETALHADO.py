# Gerador Lotofácil FINAL - DATA OBRIGATÓRIA + AsLoterias
# Autor: Jackson Leal | Parauapebas-PA | 12/01/2026
# ✅ Data nascimento OBRIGATÓRIA + Maior 18 + Nova fonte web
# OBs: Log detalhado + Fallback inteligente Gerador EM USO

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

# LOG AUTOMÁTICO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler('lotofacil.log')])

# 🎯 FIXAS: VARIÁVEIS (PRINCIPAIS) vs PERMANENTES (FALLBACK)
# ✅ dezenas_fixas = VARIÁVEIS (Baseada em Fontes Web/Arquivo)
# PRIORIDADE 1️⃣ (Dinâmica - Atualizada)

# COLETADAS das 4 fontes web ou arquivo Excel
# dezenas_fixas = coletar_estatisticas(caminho)
# Exemplo resultado: [3, 5, 9, 10, 12, 15, 20, 24] ← VARIA a cada execução

# Como funciona:
# 1. 🌐 Web scraping (4 portais) → Conta frequência → TOP 8
# 2. 📁 Arquivo Excel/CSV → Conta frequência → TOP 8
# 3. 👤 Usuário manual → Arquivo informado → TOP 8
# 4. 📊 FALLBACK (só se tudo falhar)

# Saída Real:
# 🌐 [1/4] ANALISANDO 4 PORTAIS WEB:
#    [1/4] CalculadoraOnline  ✅ 45dz → [3,5,10,11,13,20,24,25]
#    [2/4] AsLoterias         ✅ 22dz → [5,9,10,12,15,20,22,24]
#    [3/4] LotoDicas          ⚪ 0dz
#    [4/4] SomaTematica       ❌ erro

# 🎉 PORTAIS VÁLIDOS: CalculadoraOnline(45dz), AsLoterias(22dz)
# 🔥 TOP 8 ORDENADO: [3, 5, 9, 10, 12, 15, 20, 24]

# 🔒 FALLBACK_QUENTES = PERMANENTES (Reserva)
# PRIORIDADE 4️⃣ (Fixa - Plano B)
# FALLBACK_QUENTES = sorted([10, 11, 13, 14, 18, 19, 20, 25])
# SEMPRE a mesma: [10,11,13,14,18,19,20,25]

# Usado apenas quando:
# ❌ Todas 4 fontes web falham
# ❌ Nenhum arquivo Excel/CSV funciona
# ❌ Usuário não informa arquivo

# 🎲 Exemplo Execução Real
# 🔍 COLETANDO ESTATÍSTICAS...
# 🌐 [1/4] ANALISANDO 4 PORTAIS WEB:
#    [2/4] AsLoterias ✅ 18dz (2.1s)

# 🔒 FIXAS (ORDENADAS): [3, 5, 9, 10, 12, 15, 20, 24]  ← VARIÁVEL!

# JOGO 1: 03 05 09 10 12 15 20 24 01 02 06 07 17 21 22
#  ↑↑↑↑↑↑↑↑↑  ↑↑↑↑↑↑↑  (8 fixas variáveis + 7 aleatórias)

# CONFIGURAÇÕES
DEZENAS_FRIAS_PADRAO = [16, 8, 4]
TODAS_DEZENAS = list(range(1, 26))
FALLBACK_QUENTES = sorted([10, 11, 13, 14, 18, 19, 20, 25])

# ✅ 4 FONTES WEB + AsLoterias
FONTES_WEB = [
    "https://www.calculadoraonline.com.br/loterias/lotofacil",
    "https://www.somatematica.com.br/lotofacilFrequentes.php",
    "https://www.lotodicas.com.br/lotofacil/estatisticas",
    "https://www.asloterias.com.br/lotofacil/estatisticas"  # ✅ NOVA FONTE
]

NOMES_PORTAIS = ["CalculadoraOnline",
                 "SomaTematica", "LotoDicas", "AsLoterias"]


def normalizar_data_nascimento(data_input):
    """Converte DDMMYYYY → DD/MM/YYYY automaticamente."""
    data_input = data_input.strip()
    apenas_digitos = re.sub(r'[^\d]', '', data_input)
    if len(apenas_digitos) == 8 and apenas_digitos.isdigit():
        return f"{apenas_digitos[:2]}/{apenas_digitos[2:4]}/{apenas_digitos[4:]}"
    return data_input


def verificar_acesso():
    """✅ DATA OBRIGATÓRIA - Só libera maior de 18 anos."""
    print("🎯 LOTOFÁCIL 14 PONTOS - VERIFICAÇÃO OBRIGATÓRIA")
    print("=" * 70)

    nome = input("👤 Nome completo: ").strip()
    if not nome:
        print("❌ Nome obrigatório!")
        exit()

    print(f"\n🆔 {nome}, VERIFICAÇÃO DE IDADE OBRIGATÓRIA:")
    print("   📅 Aceita: 01011990 OU 01/01/1990")
    print("   ⚠️  Menores de 18 anos NÃO têm acesso!")

    while True:
        try:
            # ✅ DATA OBRIGATÓRIA - Não aceita Enter vazio
            data_raw = input("📅 DATA NASCIMENTO (OBRIGATÓRIO): ").strip()
            if not data_raw:
                print("❌ DATA OBRIGATÓRIA! Tente novamente.")
                continue

            data_normalizada = normalizar_data_nascimento(data_raw)
            print(f"📋 Data processada: {data_normalizada}")

            nascimento = datetime.strptime(data_normalizada, "%d/%m/%Y")
            hoje = datetime.now()

            # Cálculo preciso da idade
            idade = hoje.year - nascimento.year
            if (hoje.month, hoje.day) < (nascimento.month, nascimento.day):
                idade -= 1

            print(f"🎂 Idade calculada: {idade} anos")

            if idade >= 18:
                print(f"\n✅ {nome}, ACESSO LIBERADO!")
                print("   🎯 Bem-vindo ao Gerador Lotofácil Profissional!")
                logging.info(f"ACESSO LIBERADO: {nome}, {idade} anos")
                return nome
            else:
                print(f"\n❌ {nome}, ACESSO NEGADO!")
                print(f"   ⚠️  Idade insuficiente: {idade} anos (mínimo 18)")
                print("   👮 Este sistema é restrito a maiores de 18 anos.")
                input("\n🔒 Pressione Enter para encerrar...")
                logging.warning(f"ACESSO NEGADO: {nome}, {idade} anos")
                exit()

        except ValueError:
            print("❌ FORMATO INVÁLIDO!")
            print("   📋 Exemplos corretos: 01011990 OU 15/12/1985")
        except Exception as e:
            print("❌ ERRO no processamento da data!")
            logging.error(f"ERRO VERIFICAÇÃO: {e}")


def tentar_web_scraping():
    print("\n🌐 [1/4] ANALISANDO 4 PORTAIS WEB:")
    todas_freqs = Counter()
    portais_ok = []

    for i, (url, nome) in enumerate(zip(FONTES_WEB, NOMES_PORTAIS), 1):
        try:
            print(f"   [{i}/4] {nome:<15} ", end="")
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
        print(f"\n🎉 PORTAIS VÁLIDOS: {', '.join(portais_ok)}")
        print(f"🔥 TOP 8 ORDENADO: {quentes}")
        return quentes

    print("\n❌ Todas fontes web falharam")
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
    print("\n🔍 COLETANDO ESTATÍSTICAS...")
    quentes = tentar_web_scraping()
    if quentes:
        return quentes

    if caminho and os.path.exists(caminho):
        quentes = processar_arquivo_local(caminho)
        if quentes:
            return quentes

    caminho_manual = input("📁 Arquivo Excel/CSV (Enter=fixas): ").strip()
    if caminho_manual:
        quentes = processar_arquivo_local(caminho_manual)
        if quentes:
            return quentes

    print("📊 Usando fixas ordenadas 2026:", FALLBACK_QUENTES)
    return FALLBACK_QUENTES


def validar_combinacao_14pts(combinacao):
    nums = sorted(combinacao)
    soma = sum(nums)
    pares = sum(1 for n in nums if n % 2 == 0)
    setores = [0] * 5
    for n in nums:
        setores[(n-1)//5] += 1

    return (140 <= soma <= 220 and 6 <= pares <= 9 and
            sum(s >= 2 for s in setores) >= 3)


def solicitar_numero_jogos():
    while True:
        try:
            entrada = input("🎲 Nº jogos (10-20 recomendado, 0=sair): ").strip()
            n_jogos = int(entrada)

            if n_jogos == 0:
                print("\n👋 SISTEMA ENCERRADO pelo usuário!")
                print("   Até a próxima! Boa sorte nas apostas! 🎰")
                exit()

            if n_jogos < 0:
                print("❌ Número deve ser positivo!")
                continue

            return n_jogos

        except ValueError:
            print("❌ Digite apenas números!")


# === EXECUÇÃO ===
nome = verificar_acesso()
print("\n" + "="*60)

caminho = input("📁 Arquivo histórico (Enter=auto): ").strip()
dezenas_fixas = coletar_estatisticas(caminho)
dezenas_fixas = sorted(dezenas_fixas)  # Garante ordenado
print(f"\n🔒 FIXAS (ORDENADAS): {dezenas_fixas}")

n_jogos = solicitar_numero_jogos()

pool_var = [
    d for d in TODAS_DEZENAS if d not in DEZENAS_FRIAS_PADRAO and d not in dezenas_fixas]
if len(pool_var) < 10:
    pool_var = [d for d in TODAS_DEZENAS if d not in DEZENAS_FRIAS_PADRAO]

print(f"📦 Pool variáveis: {len(pool_var)} opções")
print("\n🎯 GERANDO JOGOS...")

combinacoes = set()
tentativas = 0

while len(combinacoes) < n_jogos and tentativas < 5000:
    tentativas += 1

    if len(pool_var) >= 7:
        vars7 = random.sample(pool_var, 7)
    else:
        vars7 = random.sample(TODAS_DEZENAS, 7)

    comb_temp = dezenas_fixas + vars7
    comb = sorted(list(set(comb_temp)))

    while len(comb) < 15:
        novo_num = random.choice(TODAS_DEZENAS)
        if novo_num not in comb:
            comb.append(novo_num)
        comb = sorted(comb[:15])

    if tuple(comb) not in combinacoes and validar_combinacao_14pts(comb):
        combinacoes.add(tuple(comb))

while len(combinacoes) < n_jogos:
    base = random.sample(TODAS_DEZENAS, 15)
    if tuple(base) not in combinacoes:
        combinacoes.add(tuple(base))

print(f"\n✅ {len(combinacoes)} JOGOS GERADOS!")

df_final = pd.DataFrame(list(combinacoes), columns=[
                        f'DEZ {i:02d}' for i in range(1, 16)])

print(f"\n🎰 PRÉ-VIA {nome.upper()}:")
for i, row in df_final.head().iterrows():
    jogo = [f"{int(x):02d}" for x in row]
    print(f"   JOGO {i+1:2d}: {' '.join(jogo)}")

pasta = 'C:/Users/OMEGA/OneDrive/Documentos/Jackson Leal/01 - LOTOFACIL_LOG_COMPLETO'
os.makedirs(pasta, exist_ok=True)
timestamp = datetime.now().strftime("%d%b%Y_%H%M")
arquivo = os.path.join(pasta, f'lotofacil_LOG_COMPLETO_{timestamp}.xlsx')

df_final.to_excel(arquivo, index=False, engine='openpyxl')
print(f"\n💾 EXPORTADO ({len(df_final)} jogos): {arquivo}")

print(f"\n🏆 {nome}, SUCESSO!")
print(f"💰 Custo: R$ {len(df_final)*3.50:.2f}")
print(f"   🎲 {len(df_final)} jogos gerados com estratégia profissional.")

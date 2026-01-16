# Gerador Inteligente Lotofácil com Web Scraping Automático - Versão Final
# Autor: Jackson Leal (Otimizado com asloterias.com.br)
# Parauapebas - PA | 12/01/2026
# Coleta automática + Estratégias 14 pontos

import pandas as pd
import random
from datetime import datetime
import os
from collections import Counter
import requests
from bs4 import BeautifulSoup
import time
import re

# Configuração Lotofácil otimizada
DEZENAS_FIXAS_QUENTES = [10, 11, 13, 14, 18, 19, 20, 25]
DEZENAS_FRIAS_AVOID = [16, 8]
TODAS_DEZENAS = list(range(1, 26))


def verificar_acesso():
    """Verificação de idade do usuário."""
    try:
        print("|| ACESSO AO SISTEMA ||")
        nome = input("👤 Nome: ").strip()
        if not nome:
            print("Nome obrigatório!")
            exit()

        data_nasc = input("🆔 Data nascimento (DD/MM/YYYY): ").strip()
        nascimento = datetime.strptime(data_nasc, "%d/%m/%Y")
        idade = datetime.now().year - nascimento.year - ((datetime.now().month, datetime.now().day)
         < (nascimento.month, nascimento.day))

        if idade >= 18:
            print(f"✅ {nome}, acesso liberado!")
            return nome
        print("❌ Menor de idade")
        exit()
    except:
        print("Formato inválido!")
        exit()


def coletar_historico_asloterias():
    """Coleta automática resultados Lotofácil do asloterias.com.br."""
    print("🔄 Coletando histórico automático do asloterias.com.br...")

    # URLs principais para scraping
    urls = [
        "https://asloterias.com.br/todos-resultados-lotofacil",
        "https://asloterias.com.br/lotofacil/resultados"
    ]

    historico = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Procura tabelas de resultados (padrões comuns)
            tabelas = soup.find_all('table')
            for tabela in tabelas:
                linhas = tabela.find_all('tr')
                for linha in linhas[1:]:  # Pula cabeçalho
                    cols = linha.find_all(['td', 'th'])
                    if len(cols) >= 16:  # Concurso + 15 dezenas
                        concurso = cols[0].text.strip()
                        dezenas_raw = [re.sub(r'[^\d]', '', cols[i].text)
                                       for i in range(1, 16)]
                        dezenas = [
                            int(d) for d in dezenas_raw if d.isdigit() and 1 <= int(d) <= 25]
                        if len(dezenas) == 15:
                            historico.append(
                                {'concurso': concurso, 'dezenas': sorted(dezenas)})

            time.sleep(1)  # Delay anti-ban

        except Exception as e:
            print(f"⚠️ Erro em {url}: {e}")
            continue

    # Fallback: download direto CSV se disponível
    try:
        csv_url = "https://asloterias.com.br/download-todos-resultados-lotofacil"
        df_csv = pd.read_csv(csv_url)
        print("✅ CSV direto baixado!")
        return df_csv
    except:
        pass

    if historico:
        df = pd.DataFrame(historico)
        print(f"✅ {len(df)} sorteios coletados via scraping!")
        return df
    else:
        print("❌ Falha coleta. Usando padrões fixos.")
        return pd.DataFrame()


def calcular_estatisticas(df):
    """Processa frequências das dezenas coletadas."""
    if df.empty:
        print("⚠️ Sem dados históricos. Usando quentes padrão.")
        return DEZENAS_FIXAS_QUENTES

    todas_dezenas = []
    for _, row in df.iterrows():
        if 'dezenas' in row:
            todas_dezenas.extend(row['dezenas'])

    freq = Counter(todas_dezenas)
    quentes = [num for num, _ in freq.most_common(8)]
    print(f"🔥 Dezenas mais quentes: {quentes}")
    return quentes


def validar_combinacao_14pts(combinacao):
    """Valida critérios estatísticos para 14 pontos."""
    nums = sorted(combinacao)
    soma = sum(nums)
    pares = sum(1 for n in nums if n % 2 == 0)

    # Setores: 1-5,6-10,11-15,16-20,21-25
    setores = [0] * 5
    for n in nums:
        setores[(n-1)//5] += 1

    return (150 <= soma <= 210 and
            7 <= pares <= 8 and
            all(2 <= s <= 4 for s in setores))


# INÍCIO EXECUÇÃO
nome = verificar_acesso()
print("\n🎰 GERADOR LOTOFÁCIL ESTRATÉGIA 14 PONTOS\n")

# Coleta automática histórico
df_historico = coletar_historico_asloterias()
quentes_dinamicas = calcular_estatisticas(df_historico)

# Configuração jogos
quantidade_jogos = int(input("📊 Nº jogos (10-20 recomendado): "))
dezenas_fixas = quentes_dinamicas if len(
    quentes_dinamicas) == 8 else DEZENAS_FIXAS_QUENTES
print(f"🔒 Fixas quentes: {dezenas_fixas}")

# Pool variáveis otimizado (sem frias)
pool_variaveis = [
    d for d in TODAS_DEZENAS if d not in DEZENAS_FRIAS_AVOID and d not in dezenas_fixas]

# GERAÇÃO INTELIGENTE
combinacoes = set()
tentativas = 0
max_tentativas = quantidade_jogos * 100

while len(combinacoes) < quantidade_jogos and tentativas < max_tentativas:
    tentativas += 1
    vars_7 = random.sample(pool_variaveis, 7)
    comb = sorted(dezenas_fixas + vars_7)

    if validar_combinacao_14pts(comb) and tuple(comb) not in combinacoes:
        combinacoes.add(tuple(comb))

print(f"✅ {len(combinacoes)} combinações válidas geradas!")

# DataFrame exportação
df_combs = pd.DataFrame(combinacoes, columns=[
                        f'DEZ {i:02d}' for i in range(1, 16)])

# EXPORTAÇÃO AUTOMÁTICA
pasta = 'C:/Users/OMEGA/OneDrive/Documentos/Jackson Leal/01 - LOTOFACIL_ESTRATEGIA_14PTS'
os.makedirs(pasta, exist_ok=True)
timestamp = datetime.now().strftime("%d_%b_%Y_%H%M")
arquivo = os.path.join(pasta, f'lotofacil_14pts_auto_{timestamp}.xlsx')

df_combs.to_excel(arquivo, index=False, engine='openpyxl')
print(f"\n🎉 EXPORTADO: {arquivo}")

print(f"\n📈 {nome}, {len(combinacoes)} jogos otimizados para 14 pontos!")
print("📋 Estratégia: 8 fixas quentes + 7 variáveis | Balanceados setores/pares")
print("💰 Custo 10 jogos: R$25,00 | Alta cobertura estatística")

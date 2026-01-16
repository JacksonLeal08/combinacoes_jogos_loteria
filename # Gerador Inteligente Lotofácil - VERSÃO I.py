# Gerador Lotofácil COMPLETO - DATA NASCIMENTO FLEXÍVEL
# Autor: Jackson Leal | Parauapebas-PA | 12/01/2026
# Trata DDMMYYYY ou DD/MM/YYYY automaticamente

import pandas as pd
import random
from datetime import datetime
import os
from collections import Counter
import requests
from bs4 import BeautifulSoup
import re
import time

# ========================================
# CONFIGURAÇÕES LOTOFÁCIL 2026
# ========================================
DEZENAS_FRIAS_PADRAO = [16, 8, 4]
TODAS_DEZENAS = list(range(1, 26))
FALLBACK_QUENTES = [10, 11, 13, 14, 18, 19, 20, 25]

FONTES_WEB = [
    "https://www.calculadoraonline.com.br/loterias/lotofacil",
    "https://www.somatematica.com.br/lotofacilFrequentes.php",
    "https://www.lotodicas.com.br/lotofacil/estatisticas"
]


def normalizar_data_nascimento(data_input):
    """Converte DDMMYYYY → DD/MM/YYYY automaticamente."""
    data_input = data_input.strip()

    # Remove espaços e caracteres especiais (mantém apenas dígitos)
    apenas_digitos = re.sub(r'[^\d]', '', data_input)

    if len(apenas_digitos) == 8 and apenas_digitos.isdigit():
        # Formato DDMMYYYY detectado → converte para DD/MM/YYYY
        dd = apenas_digitos[0:2]
        mm = apenas_digitos[2:4]
        yyyy = apenas_digitos[4:8]
        return f"{dd}/{mm}/{yyyy}"

    # Já está no formato correto com barras ou válido
    return data_input


def verificar_acesso():
    """Verificação idade com data flexível (DDMMYYYY ou DD/MM/YYYY)."""
    print("🎯 LOTOFÁCIL 14 PONTOS - SISTEMA INTELIGENTE")
    print("=" * 70)

    nome = input("👤 Nome: ").strip() or "Apostador"
    print(f"\n🆔 {nome}, informe data de nascimento:")
    print("   ✅ Aceita: 01011990  OU  01/01/1990")

    while True:
        try:
            data_raw = input("Data (DDMMYYYY ou DD/MM/YYYY): ").strip()
            if not data_raw:
                print("✅ Verificação pulada.")
                return nome

            # NORMALIZA AUTOMATICAMENTE
            data_normalizada = normalizar_data_nascimento(data_raw)
            print(f"📅 Data reconhecida: {data_normalizada}")

            nascimento = datetime.strptime(data_normalizada, "%d/%m/%Y")
            hoje = datetime.now()
            idade = hoje.year - nascimento.year - \
                ((hoje.month, hoje.day) < (nascimento.month, nascimento.day))

            if idade >= 18:
                print(f"✅ {nome}, maior de idade! ACESSO LIBERADO! 🎯")
                return nome
            else:
                print(f"❌ {nome}, menor de idade (idade calculada: {idade}).")
                novamente = input("Tentar novamente? (S/N): ").strip().upper()
                if novamente != 'S':
                    exit()

        except ValueError as e:
            print(f"❌ Data inválida! Exemplo: 01011990 ou 01/01/1990")
            continue


def tentar_web_scraping():
    """Coleta web com timeout rápido."""
    print("🌐 [1/3] Web scraping...")
    todas_freqs = Counter()

    for url in FONTES_WEB[:2]:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}
            response = requests.get(url, headers=headers, timeout=8)
            soup = BeautifulSoup(response.content, 'html.parser')

            for elem in soup.find_all(string=re.compile(r'\b\d{1,2}\b')):
                nums = re.findall(
                    r'\b(1[0-9]|2[0-5]|[1-9])\b', elem.parent.get_text())
                for num in nums:
                    n = int(num)
                    if 1 <= n <= 25:
                        todas_freqs[n] += 1

            time.sleep(0.8)
        except:
            continue

    if todas_freqs:
        quentes = [n for n, _ in todas_freqs.most_common(8)]
        print(f"✅ WEB: {quentes}")
        return quentes
    return None


def processar_arquivo_local(caminho):
    """Processa Excel/CSV flexível."""
    if not os.path.exists(caminho):
        return None

    try:
        print(f"📁 [2/3] Lendo {caminho}...")
        if caminho.endswith('.xlsx'):
            df = pd.read_excel(caminho)
        else:
            df = pd.read_csv(caminho)

        todas_dezenas = []
        for col in df.columns:
            if 'dezena' in col.lower() or col.startswith('DEZ'):
                col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                todas_dezenas.extend(col_data[col_data.between(1, 25)])

        if not todas_dezenas:
            for i in range(min(15, len(df.columns))):
                col_data = pd.to_numeric(
                    df.iloc[:, i], errors='coerce').dropna()
                todas_dezenas.extend(col_data[col_data.between(1, 25)])

        if todas_dezenas:
            freq = Counter(todas_dezenas.astype(int))
            quentes = [n for n, _ in freq.most_common(8)]
            print(f"✅ ARQUIVO: {quentes}")
            return quentes

    except Exception as e:
        print(f"⚠️ Erro: {e}")
    return None


def coletar_estatisticas_inteligente(caminho_opcional=""):
    """Sistema 3 níveis infalível."""
    print("\n🔍 COLETANDO ESTATÍSTICAS (3 tentativas)...")

    # Nível 1: Web
    quentes = tentar_web_scraping()
    if quentes:
        return quentes

    # Nível 2: Arquivo opcional
    if caminho_opcional and os.path.exists(caminho_opcional):
        quentes = processar_arquivo_local(caminho_opcional)
        if quentes:
            return quentes

    # Nível 3: Pergunta usuário
    caminho_manual = input("📂 Caminho Excel/CSV (Enter=fixas): ").strip()
    if caminho_manual:
        quentes = processar_arquivo_local(caminho_manual)
        if quentes:
            return quentes

    # Nível 4: Embutido
    print("📊 Usando fixas confirmadas 2026")
    return FALLBACK_QUENTES


def validar_combinacao_14pts(combinacao):
    """Validação 14 pontos profissional."""
    nums = sorted(combinacao)
    soma = sum(nums)
    pares = sum(n % 2 == 0 for n in nums)
    setores = [0] * 5
    for n in nums:
        setores[(n-1)//5] += 1

    return (150 <= soma <= 210 and 7 <= pares <= 8 and
            all(2 <= s <= 4 for s in setores))


# === INÍCIO ===
nome = verificar_acesso()
print("\n🎰 CONFIGURANDO...\n")

caminho_arquivo = input("Arquivo histórico (Enter=auto): ").strip()
dezenas_fixas = coletar_estatisticas_inteligente(caminho_arquivo)
print(f"\n🔒 FIXAS: {dezenas_fixas}")

n_jogos = int(input("🎲 Nº jogos (10-20): ") or "10")

pool_var = [
    d for d in TODAS_DEZENAS if d not in DEZENAS_FRIAS_PADRAO and d not in dezenas_fixas]

print("\n🎯 GERANDO JOGOS...")
combinacoes = set()
tentativas = 0

while len(combinacoes) < n_jogos and tentativas < 2000:
    tentativas += 1
    vars7 = random.sample(pool_var, 7)
    comb = sorted(dezenas_fixas + vars7)
    if validar_combinacao_14pts(comb) and tuple(comb) not in combinacoes:
        combinacoes.add(tuple(comb))

print(f"✅ {len(combinacoes)} jogos válidos")

# Exportação
df_final = pd.DataFrame(combinacoes, columns=[
                        f'DEZ {i:02d}' for i in range(1, 16)])
pasta = 'C:/Users/OMEGA/OneDrive/Documentos/Jackson Leal/01 - LOTOFACIL_PERFEITO'
os.makedirs(pasta, exist_ok=True)
timestamp = datetime.now().strftime("%d%b%Y_%H%M")
arquivo = os.path.join(pasta, f'lotofacil_final_{timestamp}.xlsx')

df_final.to_excel(arquivo, index=False, engine='openpyxl')
print(f"\n🎉 EXPORTADO:\n📁 {arquivo}")

print(f"\n🏆 {nome}, PRONTO!")
print(
    f"💰 Custo: R$ {len(combinacoes)*3.5:.2f} | {len(combinacoes)} jogos otimizados")
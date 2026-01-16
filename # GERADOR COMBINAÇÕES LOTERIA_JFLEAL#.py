# Nova versão do gerador de combinações para loterias
# Autor: Jackson Leal
# Parauapebas - PA
# Data: 06/01/2026


# Importação das bibliotecas necessárias
import pandas as pd
import random
from datetime import datetime
import os

# Função para verificar idade e liberar acesso ao programa


def verificar_acesso():
    try:
        # Solicita o nome do usuário
        print("|| SOLICITAÇÃO DADOS DO USUÁRIO ||")
        nome = input("👨‍🦰 Digite seu nome: ").strip()
        if not nome:
            print("O nome não pode estar vazio. Tente novamente.")
            exit()

        # Solicita a data de nascimento e ajusta o separador
        data_nascimento = input(
            "🗓️  Digite sua data de nascimento (DDMMYYYY ou DD/MM/YYYY): ").strip()
        if len(data_nascimento) == 8 and data_nascimento.isdigit():  # Formato DDMMYYYY sem separador
            data_nascimento = f"{data_nascimento[:2]}/{data_nascimento[2:4]}/{data_nascimento[4:]}"
        nascimento = datetime.strptime(data_nascimento, "%d/%m/%Y")

        # Calcula a idade
        hoje = datetime.now()
        idade = hoje.year - nascimento.year
        if (hoje.month, hoje.day) < (nascimento.month, nascimento.day):
            idade -= 1

        # Verifica se o usuário é maior de idade
        if idade >= 18:
            print(
                f"Olá, {nome}. Você é maior de idade.✅ ACESSO LIBERADO AO PROGRAMA!")
            return nome
        else:
            print(f"Olá, {nome}. Você é menor de idade.❌ ACESSO NEGADO!")
            exit()
    except ValueError:
        print("🗓️ Data de nascimento inválida. Certifique-se de usar o formato DD/MM/YYYY.")
        exit()


# Chamada da função de verificação
nome = verificar_acesso()
print()  # Pular linha

# Função para solicitar o caminho do arquivo Excel
print(" || INFORME DADOS DO ARQUIVO || ")


def solicitar_caminho_arquivo_excel():
    print("Digite o caminho completo do arquivo Excel para análise das combinações:")
    caminho_arquivo = input("Caminho do arquivo Excel: ").strip()

    # Verifica se o arquivo existe no caminho fornecido
    if not os.path.exists(caminho_arquivo):
        print(
            f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado. Tente novamente.")
        exit()

    print(f"Arquivo Excel selecionado: {caminho_arquivo}")
    return caminho_arquivo


# Solicita o caminho do arquivo Excel para análise
caminho_arquivo_excel = solicitar_caminho_arquivo_excel()

# Carrega o arquivo Excel para análise (LINHAS 73-80 CORRIGIDAS)
try:
    dados_excel = pd.read_excel(caminho_arquivo_excel)
    print("Arquivo Excel carregado com sucesso!")
    print("Pré-visualização dos dados:")
    # Formatação corrigida: sem índice, colunas alinhadas
    print(dados_excel.head().to_string(index=False))
except Exception as e:
    print(f"Erro ao carregar o arquivo Excel❗: {e}")
    exit()
print()  # Pular linha

# Escolher a modalidade de aposta
print(" || MODALIDADE JOGOS LOTERIA || ")
print(f"{nome}, escolha a modalidade de aposta:")
print("1.🍀 Mega Sena")
print("2.🍀 Lotofácil")
print("3.🍀 Lotomania")
print("4.🍀 Quina")
opcao = int(input("Digite o número correspondente a modalidade desejada: "))

# Configurações para cada modalidade
if opcao == 1:
    dezenas = list(range(1, 61))
    modalidade = 'mega_sena'
    usar_dezenas_fixas = False
elif opcao == 2:
    dezenas = list(range(1, 26))
    modalidade = 'lotofacil'
    usar_dezenas_fixas = True
elif opcao == 3:
    dezenas = list(range(0, 100))
    modalidade = 'lotomania'
    usar_dezenas_fixas = True
elif opcao == 4:
    dezenas = list(range(1, 81))
    modalidade = 'quina'
    usar_dezenas_fixas = False
else:
    print("Opção inválida!")
    exit()

# Solicitar quantidade de dezenas por combinação
dezenas_por_combinacao = int(input(
    f"Digite a quantidade de dezenas por combinação (máximo permitido: {len(dezenas)}): "))
if dezenas_por_combinacao > len(dezenas) or dezenas_por_combinacao < 1:
    print("Erro❗: Quantidade de dezenas inválida!")
    exit()

# Solicitar quantidade de jogos desejada
quantidade_jogos = int(
    input("Digite a quantidade de jogos que deseja gerar: "))
if quantidade_jogos < 1:
    print("Erro❗: A quantidade de jogos deve ser pelo menos 1!")
    exit()

# Dezenas fixas (aplicável para Lotofácil e Lotomania)
if usar_dezenas_fixas:
    print("Deseja que as dezenas fixas sejam geradas automaticamente ou informadas manualmente?")
    print("1. Automático")
    print("2. Manual")
    escolha_fixas = int(input("Digite sua escolha: "))

    if escolha_fixas == 1:
        dezenas_fixas = random.sample(dezenas, min(3, dezenas_por_combinacao))
        print(f"Dezenas Fixas (automáticas): {dezenas_fixas}")
    elif escolha_fixas == 2:
        dezenas_fixas = list(map(int, input(
            f"Informe até {min(3, dezenas_por_combinacao)} dezenas fixas: ").split()))
        dezenas_invalidas = [d for d in dezenas_fixas if d not in dezenas]
        if dezenas_invalidas:
            print(
                f"Erro❗: As dezenas {dezenas_invalidas} estão fora do intervalo permitido!")
            exit()
        if len(dezenas_fixas) > min(3, dezenas_por_combinacao):
            print("Erro❗: Mais dezenas fixas do que o permitido!")
            exit()
    else:
        print("Escolha inválida! Saindo do programa.")
        exit()
else:
    dezenas_fixas = []

# Geração de combinações
novas_combinacoes = set()
while len(novas_combinacoes) < quantidade_jogos:
    numeros_restantes = list(set(dezenas) - set(dezenas_fixas))
    selecionados = random.sample(
        numeros_restantes, dezenas_por_combinacao - len(dezenas_fixas))
    combinacao = tuple(sorted(dezenas_fixas + selecionados))
    novas_combinacoes.add(combinacao)

# Converter as combinações para DataFrame
df_novas_combinacoes = pd.DataFrame(novas_combinacoes)
df_novas_combinacoes.columns = [
    f"DEZ {i:02}" for i in range(1, dezenas_por_combinacao + 1)]

# Obter o caminho de exportação
output_folder = 'C:/Users/OMEGA/OneDrive/Documentos/Jackson Leal/01 - COMBINAES_MEGA_VIRADA_PRIMOS'
os.makedirs(output_folder, exist_ok=True)
data_atual = datetime.now().strftime("%d%b%Y").lower()

# EXPORTAÇÃO SOMENTE EXCEL (PDF REMOVIDO)
print("Gerando arquivo Excel...")
output_path = os.path.join(
    output_folder, f'{modalidade}_{data_atual}.xlsx')
if os.path.exists(output_path):
    escolha = input(
        f"O arquivo '{output_path}' já existe. Deseja sobrescrever (S/N) ou informar outro nome? ").strip().lower()
    if escolha == 'n':
        novo_nome = input(
            "Informe o novo nome para o arquivo (sem extensão): ").strip()
        output_path = os.path.join(output_folder, f'{novo_nome}.xlsx')

df_novas_combinacoes.to_excel(
    output_path, index=False, header=True, engine='openpyxl')
print(f"Arquivo Excel exportado para: {output_path}")

print()  # Pular linha
# Finalização
print(f'{nome}, geração de combinações concluída com sucesso, verificar na pasta de destino "COMBINAÇÕES"!')

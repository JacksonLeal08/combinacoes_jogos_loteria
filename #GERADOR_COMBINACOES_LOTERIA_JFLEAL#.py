# Atualizando corpo do código
# Jackson Leal, 04/02/2025

# Importação das bibliotecas necessárias
import pandas as pd
import random
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.units import inch
import os

# Função para verificar idade e liberar acesso ao programa
def verificar_acesso():
    try:
        # Solicita o nome do usuário
        print("|| SOLICITAÇÃO DADOS DO USUÁRIO ||")
        nome = input(f"🫵  Digite seu nome: ").strip()
        if not nome:
            print(f"⛔ O nome não pode estar vazio. Tente novamente.")
            exit()

        # Solicita a data de nascimento e ajusta o separador
        data_nascimento = input(f"🗓️  Digite sua data de nascimento (DDMMYYYY ou DD/MM/YYYY): ").strip()
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
            print(f"Olá, {nome}. Você é maior de idade.✅  ACESSO LIBERADO AO PROGRAMA!")
            return nome
        else:
            print(f"Olá, {nome}. Você é menor de idade.❌ ACESSO NEGADO!")
            exit()
    except ValueError:
        print(f"❌ Data de nascimento inválida. Certifique-se de usar o formato DD/MM/YYYY.")
        exit()

# Chamada da função de verificação
nome = verificar_acesso()
print() # Pular linha
# Função para solicitar o caminho do arquivo Excel
print(" || 🗃️  INFORME DADOS DO ARQUIVO 🗃️  || ")
def solicitar_caminho_arquivo_excel():
    print(f"📂 Digite o caminho completo do arquivo Excel para análise das combinações:")
    caminho_arquivo = input("Caminho do arquivo Excel: ").strip()

    # Verifica se o arquivo existe no caminho fornecido
    if not os.path.exists(caminho_arquivo):
        print(f"❌ Erro: O arquivo '{caminho_arquivo}' não foi encontrado. Tente novamente.")
        exit()
    
    print(f"📂 Arquivo Excel selecionado: {caminho_arquivo}")
    return caminho_arquivo

# Solicita o caminho do arquivo Excel para análise
caminho_arquivo_excel = solicitar_caminho_arquivo_excel()

# Carrega o arquivo Excel para análise
try:
    dados_excel = pd.read_excel(caminho_arquivo_excel)
    print("✅ Arquivo Excel carregado com sucesso!")
    print(f"📋 Pré-visualização dos dados:\n{dados_excel.head()}")
except Exception as e:
    print(f"❌ Erro ao carregar o arquivo Excel: {e}")
    exit()
print() # Pular linha

# Escolher a modalidade de aposta
print(" || MODALIDADE JOGOS LOTERIA || ")
print(f"{nome}, escolha a modalidade de aposta:")
print(f"🍀 1. Mega Sena")
print(f"🍀 2. Lotofácil")
print(f"🍀 3. Lotomania")
print(f"🍀 4. Quina")
opcao = int(input("Digite o número da modalidade desejada: "))

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
    print(f"❌ Opção inválida!")
    exit()

# Solicitar quantidade de dezenas por combinação
dezenas_por_combinacao = int(input(f"Digite a quantidade de dezenas por combinação (máximo permitido: {len(dezenas)}): "))
if dezenas_por_combinacao > len(dezenas) or dezenas_por_combinacao < 1:
    print(f"❌ Erro: Quantidade de dezenas inválida!")
    exit()

# Solicitar quantidade de jogos desejada
quantidade_jogos = int(input("Digite a quantidade de jogos que deseja gerar: "))
if quantidade_jogos < 1:
    print(f"❌ Erro: A quantidade de jogos deve ser pelo menos 1!")
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
        dezenas_fixas = list(map(int, input(f"Informe até {min(3, dezenas_por_combinacao)} dezenas fixas: ").split()))
        dezenas_invalidas = [d for d in dezenas_fixas if d not in dezenas]
        if dezenas_invalidas:
            print(f"❌ Erro: As dezenas {dezenas_invalidas} estão fora do intervalo permitido!")
            exit()
        if len(dezenas_fixas) > min(3, dezenas_por_combinacao):
            print("❌ Erro: Mais dezenas fixas do que o permitido!")
            exit()
    else:
        print(f"⛔ Escolha inválida! Saindo do programa.")
        exit()
else:
    dezenas_fixas = []

# Geração de combinações (continuar lógica já implementada...)
novas_combinacoes = set()
while len(novas_combinacoes) < quantidade_jogos:
    numeros_restantes = list(set(dezenas) - set(dezenas_fixas))
    selecionados = random.sample(numeros_restantes, dezenas_por_combinacao - len(dezenas_fixas))
    combinacao = tuple(sorted(dezenas_fixas + selecionados))
    novas_combinacoes.add(combinacao)

# Converter as combinações para DataFrame
df_novas_combinacoes = pd.DataFrame(novas_combinacoes)
df_novas_combinacoes.columns = [f"DEZ {i:02}" for i in range(1, dezenas_por_combinacao + 1)]

# Obter o caminho de exportação
output_folder = 'C:/Users/jacks/Documents/_Cursos_Jackson_Leal/00 - Jogos_Loteria_Caixa/02 - COMBINACOES/'
os.makedirs(output_folder, exist_ok=True)
data_atual = datetime.now().strftime("%d%b%Y").lower()

# Escolher o formato de exportação
print("Escolha o formato de exportação:")
print("1. Excel")
print("2. PDF")
formato_opcao = int(input("Digite o número do formato desejado: "))

# Gerar arquivo de exportação
if formato_opcao == 1:
    # Exportar como Excel
    output_path = os.path.join(output_folder, f'{modalidade}_{data_atual}.xlsx')
    if os.path.exists(output_path):
        escolha = input(f"📂 O arquivo '{output_path}' já existe. Deseja sobrescrever (S/N) ou informar outro nome? ").strip().lower()
        if escolha == 'n':
            novo_nome = input("Informe o novo nome para o arquivo (sem extensão): ").strip()
            output_path = os.path.join(output_folder, f'{novo_nome}.xlsx')
    df_novas_combinacoes.to_excel(output_path, index=False, header=True, engine='openpyxl')
    print(f"✅ Arquivo Excel exportado para: {output_path}")
elif formato_opcao == 2:
    # Exportar como PDF
    output_path = os.path.join(output_folder, f'{modalidade}_{data_atual}.pdf')
    if os.path.exists(output_path):
        escolha = input(f"📂 O arquivo '{output_path}' já existe. Deseja sobrescrever (S/N) ou informar outro nome? ").strip().lower()
        if escolha == 'n':
            novo_nome = input("📂 Informe o novo nome para o arquivo (sem extensão): ").strip()
            output_path = os.path.join(output_folder, f'{novo_nome}.pdf')
    pdf = SimpleDocTemplate(output_path, pagesize=A4)
    data = [df_novas_combinacoes.columns.tolist()] + df_novas_combinacoes.values.tolist()
    table = Table(data, colWidths=[0.7 * inch] * dezenas_por_combinacao)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    pdf.build([table])
    print(f"✅ Arquivo PDF exportado para: {output_path}")
else:
    print("Opção de formato inválida!")
print() # Pular linha
# Finalização
print(f'{nome}, ✅ geração de combinações concluída com sucesso, verificar na pasta de destino "COMBINAÇÕES"!')

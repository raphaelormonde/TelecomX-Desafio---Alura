import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

url = "https://raw.githubusercontent.com/ingridcristh/challenge2-data-science/refs/heads/main/TelecomX_Data.json"


# ==========================================
# EXTRAÇÃO
# ==========================================


# requisição para obter os dados
response = requests.get(url)

# Ver se a requisição deu certo e carregar os dados
if response.status_code == 200:
    dados_json = response.json()
    
    # Converter o JSON para um DataFrame do Pandas
    df = pd.DataFrame(dados_json)
    
    print("Dados carregados com sucesso!")
    # Visualizar as 5 primeiras linhas para conferir
    print(df.head())
else:
    print(f"Erro ao acessar a API: {response.status_code}")


# ==========================================
# CONHECENDO O DATASET
# ==========================================


# 1. Expandir os dados aninhados (essencial para ver as colunas do dicionário)
df_final = pd.json_normalize(dados_json)

# 2. Verificar o resumo técnico (conforme a dica da imagem: df.info)
print("--- Informações Gerais ---")
df_final.info()

# 3. Verificar apenas os tipos de dados (conforme a dica da imagem: df.dtypes)
print("\n--- Tipos de Dados ---")
print(df_final.dtypes)


# ==========================================
# VERIFICANDO INCONSISTÊNCIAS
# ==========================================


# 1. Verificar valores únicos nas colunas categóricas (usando pandas.unique)
# Isso ajuda a identificar inconsistências como 'Sim', 'sim', 'Yes' na mesma coluna.
print("--- Valores únicos por coluna ---")
colunas_categoricas = ['Churn', 'account.Contract', 'account.PaymentMethod']

for col in colunas_categoricas:
    print(f"Valores únicos em {col}: {df_final[col].unique()}")

# 2. Verificar se há valores ausentes (Nulos)
print("\n--- Quantidade de valores nulos ---")
print(df_final.isnull().sum())

# 3. Verificar se existem linhas duplicadas
duplicados = df_final.duplicated().sum()
print(f"\nTotal de linhas duplicadas: {duplicados}")


# ==========================================
# TRATANDO INCONSISTÊNCIAS
# ==========================================




# 1. Tratar a coluna Churn: Remover linhas onde o valor está vazio ('')
# Como o Churn é o nosso alvo, não podemos ter dados vazios aqui.
df_final = df_final[df_final['Churn'] != '']

# 2. Converter strings vazias em outras colunas para NaN (nulo oficial do Pandas)
# Isso permite que o Pandas reconheça campos vazios como nulos reais.
df_final = df_final.replace('', np.nan)

# 3. Corrigir a coluna account.Charges.Total
# Agora que os vazios foram tratados, podemos converter para numérico.
df_final['account.Charges.Total'] = pd.to_numeric(df_final['account.Charges.Total'], errors='coerce')

# 4. Remover nulos remanescentes
# Vamos verificar quantos nulos reais temos agora
print("\n--- Verificação após correções ---")
print(df_final.isnull().sum())

# 5. Resetar o índice após remover linhas para manter a organização
df_final.reset_index(drop=True, inplace=True)

print(f"\nDados limpos! Novo total de registos: {len(df_final)}")


# ==========================================
# VERIFICAÇÃO
# ==========================================


# Verifica se o valor '' (vazio) sumiu
print("Valores únicos no Churn:", df_final['Churn'].unique())
# O resultado esperado é apenas: ['No', 'Yes']

# Verifica se o Dtype mudou para float64
print("Tipo da coluna Total:", df_final['account.Charges.Total'].dtype)
# O resultado esperado é: float64



# Mostra onde ainda existem buracos nos dados
sns.heatmap(df_final.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Verificação de Dados Faltantes')
plt.show()


# ==========================================
# ANÁLISE DESCRITIVA
# ==========================================


# 1. Gerar estatísticas descritivas para as colunas numéricas
# O describe() calcula automaticamente média, desvio padrão, min, max e quartis.
print("--- Estatísticas Descritivas das Variáveis Numéricas ---")
estatisticas = df_final.describe()
print(estatisticas)

# 2. Analisar especificamente a média e mediana de gastos
media_mensal = df_final['account.Charges.Monthly'].mean()
mediana_mensal = df_final['account.Charges.Monthly'].median()

print(f"\nMédia de Gastos Mensais: R$ {media_mensal:.2f}")
print(f"Mediana de Gastos Mensais: R$ {mediana_mensal:.2f}")

# 3. Verificar estatísticas separadas por quem saiu 
print("\n--- Média de Gastos Mensais por Status de Churn ---")
print(df_final.groupby('Churn')['account.Charges.Monthly'].mean())

# Clientes que cancelam o serviço gastam, em média, R$ 13,18 a mais por mês do que os que permanecem. 
# Isso sugere que o preço alto pode ser um dos principais motivadores da evasão.


# ==============================================================================
# DISTRIBUIÇÃO DA EVASÃO
# ==============================================================================


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))

# Criando o gráfico de contagem 
ax = sns.countplot(x='Churn', data=df_final, palette='viridis')

# Adicionando a porcentagem em cima das barras para facilitar a leitura
total = len(df_final)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2 - 0.05
    y = p.get_height() + 50
    ax.annotate(percentage, (x, y), fontsize=12, fontweight='bold')

plt.title('Distribuição da Evasão de Clientes (Churn)', fontsize=14)
plt.xlabel('O cliente saiu?', fontsize=12)
plt.ylabel('Quantidade de Clientes', fontsize=12)

plt.show()


# ==============================================================================
# CONTAGEM DE EVASÃO POR VARIÁVEIS CATEGÓRICAS
# ==============================================================================




# Lista de variáveis sugeridas pela imagem para análise
colunas_analise = ['customer.gender', 'account.Contract', 'account.PaymentMethod']

# Criando a estrutura de subplots
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

for i, col in enumerate(colunas_analise):
    # Criando o gráfico de contagem agrupado por Churn
    sns.countplot(x=col, hue='Churn', data=df_final, ax=axes[i], palette='viridis')
    
    # Ajustes estéticos
    axes[i].set_title(f'Churn por {col}', fontsize=14)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Quantidade de Clientes')
    
    # Rotacionar legendas no eixo X para o Método de Pagamento (que tem nomes longos)
    if col == 'account.PaymentMethod':
        axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# ==============================================================================
# CONTAGEM DE EVASÃO POR VARIÁVEIS NUMÉRICAS
# ==============================================================================


import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 1. Relação entre Tempo de Casa (Tenure) e Churn
# Isso responde: Clientes novos saem mais que os antigos?
sns.boxplot(x='Churn', y='customer.tenure', data=df_final, ax=axes[0], palette='viridis')
axes[0].set_title('Tempo de Casa (Meses) vs Evasão')

# 2. Relação entre Gastos Mensais e Churn
# Isso responde: Clientes que pagam mais caro saem mais?
sns.boxplot(x='Churn', y='account.Charges.Monthly', data=df_final, ax=axes[1], palette='magma')
axes[1].set_title('Gastos Mensais vs Evasão')

plt.tight_layout()
plt.show()


# ==============================================================================
# SALVANDO CSV
# ==============================================================================


df_final.to_csv('TelecomX.csv', index=False)
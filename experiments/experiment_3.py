import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração de diretórios e carregamento de dados
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(root_dir, 'config.json')

try:
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print(f"Error: Configuration file not found at {config_path}")
    sys.exit(1)

dataset_path = os.path.expanduser(config.get('dataset_path', './PAD-UFES-20'))
metadata_path = os.path.join(dataset_path, 'metadata.csv')
results_path = os.path.join(root_dir, 'results', 'metadata_exploratory_analysis')
os.makedirs(results_path, exist_ok=True)

metadata = pd.read_csv(metadata_path)
# Padroniza os valores booleanos
data = metadata.replace({'True': 'YES', 'False': 'NO', True: 'YES', False: 'NO'})

# =======================
# Impressão de análises textuais
# =======================
print("=== Preview dos Dados ===")
print(data.head(), "\n")

# 1. Distribuição de Idade por Gênero
print("=== Distribuição de Idade por Gênero ===")
all_age_mean = data['age'].mean()
all_age_median = data['age'].median()
male_age_mean = data[data['gender'] == 'MALE']['age'].mean()
male_age_median = data[data['gender'] == 'MALE']['age'].median()
female_age_mean = data[data['gender'] == 'FEMALE']['age'].mean()
female_age_median = data[data['gender'] == 'FEMALE']['age'].median()

print(f"Média geral de idade: {all_age_mean:.2f} anos | Mediana: {all_age_median} anos")
print(f"Média de idade (Masculino): {male_age_mean:.2f} anos | Mediana: {male_age_median} anos")
print(f"Média de idade (Feminino): {female_age_mean:.2f} anos | Mediana: {female_age_median} anos\n")

# 2. Estatísticas de Idade por Diagnóstico
print("=== Estatísticas de Idade por Diagnóstico ===")
diagnostics = data['diagnostic'].unique()
for diag in diagnostics:
    diag_data = data[data['diagnostic'] == diag]['age']
    print(f"Diagnóstico {diag}:")
    print(f"  Mínimo: {diag_data.min()} anos")
    print(f"  Máximo: {diag_data.max()} anos")
    print(f"  Média: {diag_data.mean():.2f} anos")
    print(f"  Mediana: {diag_data.median()} anos")
    print(f"  1º Quartil: {diag_data.quantile(0.25)} anos")
    print(f"  3º Quartil: {diag_data.quantile(0.75)} anos\n")

# 3. Distribuição de Gênero por Diagnóstico
print("=== Distribuição de Gênero por Diagnóstico ===")
gender_diag = data.groupby(['diagnostic', 'gender']).size().unstack(fill_value=0)
print(gender_diag, "\n")

# 4. Distribuição da Escala de Fitzpatrick por Diagnóstico
print("=== Distribuição da Escala de Fitzpatrick por Diagnóstico ===")
if 'fitspatrick' in data.columns:
    fitz_diag = data.groupby(['diagnostic', 'fitspatrick']).size().unstack(fill_value=0)
    print(fitz_diag, "\n")
else:
    print("Coluna 'fitspatrick' não encontrada.\n")

# 5. Frequência de Regiões Anatômicas
print("=== Frequência de Regiões Anatômicas ===")
if 'region' in data.columns:
    region_counts = data['region'].value_counts()
    print(region_counts, "\n")
else:
    print("Coluna 'region' não encontrada.\n")

# 6. Análise de Histórico Familiar
print("=== Histórico Familiar ===")
for parent, label in [('father', 'Pai'), ('mother', 'Mãe')]:
    col_name = f'background_{parent}'
    if col_name in data.columns:
        bg_counts = data[col_name].value_counts()
        print(f"Histórico {label}:")
        print(bg_counts, "\n")
    else:
        print(f"Coluna {col_name} não encontrada.\n")

# 7. Análise de Diâmetros
print("=== Análise de Diâmetros ===")
diam_cols = ['diameter_1', 'diameter_2', 'diagnostic']
if all(col in data.columns for col in diam_cols):
    diam_data = data[diam_cols].dropna()
    print("Estatísticas descritivas para os diâmetros (por diagnóstico):")
    for diag in diagnostics:
        diag_diam = diam_data[diam_data['diagnostic'] == diag]
        if not diag_diam.empty:
            print(f"\nDiagnóstico {diag}:")
            print(diag_diam[['diameter_1', 'diameter_2']].describe())
else:
    print("Colunas de diâmetro não encontradas no dataset.\n")

# 8. Análise de Características Booleanas
print("=== Análise de Características Booleanas por Diagnóstico ===")
bool_features = [
    'smoke', 'drink', 'pesticide', 'skin_cancer_history', 'cancer_history',
    'has_piped_water', 'has_sewage_system', 'itch', 'grew', 'hurt',
    'changed', 'bleed', 'elevation'
]
for feat in bool_features:
    if feat in data.columns:
        print(f"\nAnálise da variável '{feat}':")
        for diag in diagnostics:
            diag_data = data[data['diagnostic'] == diag]
            counts = diag_data[feat].value_counts()
            counts = counts.reindex(['YES', 'NO', 'UNK'], fill_value=0)
            print(f"  Diagnóstico {diag} -> YES: {counts['YES']}, NO: {counts['NO']}, UNK: {counts['UNK']}")
    else:
        print(f"Coluna '{feat}' não encontrada.")


# =======================
# Geração e Salvamento dos Gráficos
# =======================

# 1. Distribuição de Idade por Gênero
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], color="lime", label='Todos', kde=True, bins=15)
sns.histplot(data[data['gender'] == 'MALE']['age'], color="navy", label='Masculino', kde=True, bins=15)
sns.histplot(data[data['gender'] == 'FEMALE']['age'], color="coral", label='Feminino', kde=True, bins=15)
plt.grid(color='black', linestyle='dotted', linewidth=0.7)
plt.xlabel("Idade")
plt.legend()
plt.savefig(os.path.join(results_path, 'distribuicao_idade.png'), dpi=200)
plt.close()

# 2. Boxplot de Idade por Diagnóstico
plt.figure(figsize=(12, 6))
sns.boxplot(y='age', x='diagnostic', data=data, palette="Blues_d")
plt.grid(color='black', linestyle='dotted', linewidth=0.7)
plt.xlabel("Diagnóstico")
plt.ylabel("Idade")
plt.savefig(os.path.join(results_path, 'boxplot_idade_diagnostico.png'), dpi=200)
plt.close()

# 3. Distribuição de Gênero por Diagnóstico
plt.figure(figsize=(12, 6))
sns.countplot(x="diagnostic", hue="gender", data=data, palette="Blues_d")
plt.xlabel("Diagnóstico")
plt.ylabel("Contagem")
plt.savefig(os.path.join(results_path, 'genero_por_diagnostico.png'), dpi=200)
plt.close()

# 4. Distribuição da Escala de Fitzpatrick por Diagnóstico
plt.figure(figsize=(12, 6))
if 'fitspatrick' in data.columns:
    sns.countplot(x="diagnostic", hue="fitspatrick", data=data)
    plt.legend(loc='upper right', title="Fitzpatrick")
    plt.xlabel("Diagnóstico")
    plt.ylabel("Contagem")
    plt.savefig(os.path.join(results_path, 'distribuicao_fitzpatrick.png'), dpi=200)
    plt.close()
else:
    print("Coluna 'fitspatrick' não encontrada para gerar gráfico.")

# 5. Análise de Regiões Anatômicas
plt.figure(figsize=(12, 6))
if 'region' in data.columns:
    sns.countplot(x="diagnostic", hue="region", data=data)
    plt.legend(loc='right', prop={'size': 7}, title="Região")
    plt.xlabel("Diagnóstico")
    plt.ylabel("Contagem")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'regioes_por_diagnostico.png'), dpi=200)
    plt.close()
else:
    print("Coluna 'region' não encontrada para gerar gráfico.")

# 6. Frequência das Regiões Anatômicas
plt.figure(figsize=(10, 8))
if 'region' in data.columns:
    region_counts = data.groupby(['region']).size().sort_values(ascending=False)
    sns.barplot(x=region_counts.values, y=region_counts.index, palette="Blues_d", orient='h')
    plt.grid(color='black', linestyle='dotted', linewidth=0.7)
    plt.xlabel("Frequência")
    plt.ylabel("Região Anatômica")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'frequencia_regioes.png'), dpi=200)
    plt.close()
else:
    print("Coluna 'region' não encontrada para gerar gráfico.")

# 7. Análise de Histórico Familiar
for parent, pt_label in [('father', 'Pai'), ('mother', 'Mãe')]:
    plt.figure(figsize=(10, 6))
    col_name = f'background_{parent}'
    if col_name in data.columns:
        bg_counts = data.groupby([col_name]).size().sort_values(ascending=False)
        sns.barplot(x=bg_counts.values, y=bg_counts.index, palette="Blues_d", orient='h')
        plt.grid(color='black', linestyle='dotted', linewidth=0.7)
        plt.xlabel("Frequência")
        plt.ylabel(f"Histórico {pt_label}")
        plt.savefig(os.path.join(results_path, f'historico_familiar_{parent}.png'), dpi=200)
        plt.close()
    else:
        print(f"Coluna {col_name} não encontrada para gerar gráfico.")

# 8. Análise de Diâmetros
diam_cols = ['diameter_1', 'diameter_2', 'diagnostic']
if all(col in data.columns for col in diam_cols):
    diam_data = data[diam_cols].dropna()
    g = sns.pairplot(diam_data, hue="diagnostic")
    g.fig.suptitle("Relação entre Diâmetros por Diagnóstico", y=1.02)
    g.savefig(os.path.join(results_path, 'analise_diametros.png'), dpi=200)
    plt.close()
else:
    print("Colunas de diâmetro não encontradas no dataset para gerar gráfico.")

# 9. Análise de Características Booleanas com Gráficos de Pizza e de Contagem
bool_features = [
    'smoke', 'drink', 'pesticide', 'skin_cancer_history', 'cancer_history',
    'has_piped_water', 'has_sewage_system', 'itch', 'grew', 'hurt',
    'changed', 'bleed', 'elevation'
]

bool_translations = {
    'smoke': 'Fumante',
    'drink': 'Consumo de Álcool',
    'pesticide': 'Exposição a Pesticidas',
    'skin_cancer_history': 'Histórico de Câncer de Pele',
    'cancer_history': 'Histórico de Câncer',
    'has_piped_water': 'Água Encanada',
    'has_sewage_system': 'Esgoto',
    'itch': 'Coceira',
    'grew': 'Crescimento',
    'hurt': 'Dor',
    'changed': 'Mudanças',
    'bleed': 'Sangramento',
    'elevation': 'Elevação'
}

colors = {'YES': '#4CAF50', 'NO': '#F44336', 'UNK': '#9E9E9E'}

def create_pie_chart(data, feature, diagnostic, translation):
    plt.figure(figsize=(8, 8))
    diagnostic_data = data[data['diagnostic'] == diagnostic]
    counts = diagnostic_data[feature].value_counts()
    counts = counts.reindex(['YES', 'NO', 'UNK'], fill_value=0)
    patches, texts, autotexts = plt.pie(
        counts,
        labels=counts.index,
        colors=[colors[x] for x in counts.index],
        autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
        startangle=90,
        textprops={'fontsize': 12}
    )
    plt.setp(autotexts, color='white', weight='bold')
    plt.title(f"{translation[feature]} - {diagnostic}", fontsize=14, pad=20)
    plt.savefig(os.path.join(results_path, f'pie_{feature}_{diagnostic}.png'), dpi=200, bbox_inches='tight')
    plt.close()

for feat in bool_features:
    if feat in data.columns:
        for diag in diagnostics:
            create_pie_chart(data, feat, diag, bool_translations)
        plt.figure(figsize=(12, 6))
        sns.countplot(x="diagnostic", hue=feat, data=data, palette="Blues_d")
        plt.legend(loc='upper right', title=bool_translations[feat])
        plt.xlabel("Diagnóstico")
        plt.ylabel("Contagem")
        plt.savefig(os.path.join(results_path, f'distribuicao_unica_{feat}.png'), dpi=200)
        plt.close()
    else:
        print(f"Coluna '{feat}' não encontrada para gerar gráficos booleanos.")

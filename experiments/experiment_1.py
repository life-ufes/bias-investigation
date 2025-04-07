#Experiment 1 - Age Impact on Cancer Classification Probabilities

import sys
import os
import json
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# Obter o caminho absoluto do diretório raiz do projeto (neste caso, bias-investigation)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)  # Insere o caminho do diretório raiz no início da lista de caminhos

# Caminho correto para o arquivo de configuração (dentro da raiz do projeto)
config_path = os.path.join(root_dir, 'config.json')

# Carregar o arquivo de configuração
try:
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print(f"Erro: Arquivo de configuração não encontrado em {config_path}")
    sys.exit(1)

# Caminhos definidos no arquivo de configuração
dataset_path = config.get('dataset_path', './PAD-UFES-20')
metadata_path = os.path.join(dataset_path, 'metadata.csv')
images_path = os.path.join(dataset_path, 'images')
results_path = os.path.join(root_dir, 'results', 'cancer_predictions_by_age.csv')

# Importar as funções utilitárias
sys.path.insert(0, os.path.join(root_dir, 'utils'))
from model_inference import set_and_load_model, make_inference
from data_loader import load_metadata, load_image
from metadata_mapper import map_metadata

# Verificar se o resultado já existe
if os.path.exists(results_path):
    # Carregar os resultados existentes
    df = pd.read_csv(results_path)
else:
    # Carregar o modelo
    MODEL = set_and_load_model()

    # Carregar metadados
    metadata = load_metadata(metadata_path)

    # Filtrar para lesões de melanoma, BCC e SCC
    melanoma_metadata = metadata[metadata['diagnostic'].isin(['MEL', 'BCC', 'SCC'])]

    # Inicializar uma lista para armazenar resultados
    results = []

    # Faixa de idade possível
    age_range = range(10, 101)

    # Iterar sobre cada lesão de melanoma, BCC e SCC
    for _, row in melanoma_metadata.iterrows():
        img_id = row['img_id']
        img_path = os.path.join(images_path, img_id)
        
        if not os.path.exists(img_path):
            continue  # Pular se o arquivo de imagem não existir
        
        img = load_image(images_path, img_id)
        
        # Converter metadados
        mapped_metadata = map_metadata(row)

        for age in age_range:
            # Atualizar a idade no mapeamento de metadados
            mapped_metadata['age'] = age

            # Fazer a inferência
            pred, pred_label, pred_prob = make_inference(
                MODEL, img, mapped_metadata['region'], mapped_metadata['fitzpatrick'], mapped_metadata['age'],
                mapped_metadata['gender'], mapped_metadata['smoke'], mapped_metadata['drink'], mapped_metadata['itch'],
                mapped_metadata['grew'], mapped_metadata['bleed'], mapped_metadata['hurt'], mapped_metadata['changed'],
                mapped_metadata['elevation'], mapped_metadata['cancer_history'], mapped_metadata['skin_cancer_history'], verbose=False
            )
            
            # Armazenar os resultados
            results.append({
                "Age": age,
                "Diagnosis": row['diagnostic'],
                "ACK": pred[0],
                "BCC": pred[1],
                "MEL": pred[2],
                "NEV": pred[3],
                "SCC": pred[4],
                "SEK": pred[5],
                "Predicted_Label": pred_label,
                "Predicted_Probability": pred_prob
            })

    # Converter resultados para DataFrame
    df = pd.DataFrame(results)

    # Salvar DataFrame em CSV para referência futura
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.to_csv(results_path, index=False)

translations = {
    'diagnosis': {
        'MEL': 'Melanoma',
        'BCC': 'Carcinoma Basocelular',
        'SCC': 'Carcinoma Espinocelular'
    },
    'labels': {
        'age': 'Idade',
        'probability': 'Probabilidade (%)',
        'class': 'Classe',
        'combined_title': 'Impacto da Idade nas Probabilidades de Classificação para {} - Visão Agregada',
        'mosaic_title': 'Impacto da Idade nas Probabilidades de Classificação para {} - Visões Individuais',
        'individual_title': 'Probabilidade de {} vs Idade para Casos de {}'
    }
}

# Criar diretório para os plots
plots_dir = os.path.join(os.path.dirname(results_path), 'age_impact_plots_pt')
os.makedirs(plots_dir, exist_ok=True)

for diagnosis in ['MEL', 'BCC', 'SCC']:
    diagnosis_df = df[df['Diagnosis'] == diagnosis]
    melted_df = diagnosis_df.melt(id_vars='Age', value_vars=['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK'],
                                  var_name='Class', value_name='Probability')
    melted_df['Probability'] *= 100

    # Plot Combinado
    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(data=melted_df, x='Age', y='Probability', hue='Class', ci='sd')
    plt.xlabel(translations['labels']['age'])
    plt.ylabel(translations['labels']['probability'])
    plt.title(translations['labels']['combined_title'].format(translations['diagnosis'][diagnosis]))
    plt.grid(True)
    plt.legend(title=translations['labels']['class'], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'combinado_{diagnosis}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot em Mosaico
    classes = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(translations['labels']['mosaic_title'].format(translations['diagnosis'][diagnosis]), 
                y=1.02, fontsize=14)
    
    for i, class_name in enumerate(classes, 1):
        ax = plt.subplot(2, 3, i)
        class_df = melted_df[melted_df['Class'] == class_name]
        
        if not class_df.empty:
            sns.lineplot(data=class_df, x='Age', y='Probability', ci='sd',
                        color=sns.color_palette()[i-1], legend=False, ax=ax)
            ax.set(title=class_name, 
                  xlabel=translations['labels']['age'],
                  ylabel=translations['labels']['probability'],
                  ylim=(-5, 105))
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'Sem Dados', ha='center', va='center')
            ax.set(title=class_name)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'mosaico_{diagnosis}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plots Individuais
    for class_name in classes:
        plt.figure(figsize=(8, 5))
        class_df = melted_df[melted_df['Class'] == class_name]
        
        if not class_df.empty:
            sns.lineplot(data=class_df, x='Age', y='Probability', ci='sd',
                        color=sns.color_palette()[classes.index(class_name)])
            plt.title(translations['labels']['individual_title'].format(
                class_name, 
                translations['diagnosis'][diagnosis]
            ))
            plt.xlabel(translations['labels']['age'])
            plt.ylabel(translations['labels']['probability'])
            plt.grid(True)
            plt.ylim(-5, 105)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{diagnosis}_{class_name}_individual.png'), dpi=300)
            plt.close()

print(f"Gráficos salvos em: {plots_dir}")
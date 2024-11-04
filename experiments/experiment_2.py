# Experiment 2 - Individual Binary Metadata Impact on Cancer Classification Probabilities

import os
import sys
import json
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

# Obter o caminho absoluto do diretório raiz do projeto (neste caso, bias-investigation)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root_dir, 'utils'))
from model_inference import set_and_load_model, make_inference
from metadata_mapper import map_metadata

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
dataset_path = os.path.expanduser(config.get('dataset_path', './PAD-UFES-20'))
metadata_path = os.path.join(dataset_path, 'metadata.csv')
images_path = os.path.join(dataset_path, 'images')
results_path = os.path.join(root_dir, 'results', 'individual_metadata_impact.csv')

# Lista de atributos de metadados para analisar
attributes = ['bleed', 'cancer_history', 'changed', 'drink', 'elevation', 
              'grew', 'hurt', 'itch', 'skin_cancer_history', 'smoke']

# Se o arquivo de resultados já existir, apenas plote os gráficos
if os.path.exists(results_path):
    df = pd.read_csv(results_path)

    # Verificar se a coluna 'diagnostic' está presente
    if 'diagnostic' not in df.columns:
        print("Error: 'diagnostic' column not found in the DataFrame.")
        sys.exit(1)

    # Melhor visualização para impactos individuais
    diagnostics = ['MEL', 'BCC', 'SCC']
    for diagnosis in diagnostics:
        plt.figure(figsize=(20, 20))

        diagnosis_df = df[df['diagnostic'] == diagnosis]
        print(diagnosis_df)

        # Iterar sobre cada atributo de metadados para criar subplots separados
        for i, attribute in enumerate(attributes, 1):
            plt.subplot(4, 3, i)

            # Extrair dados originais e invertidos para o atributo atual
            original_data = diagnosis_df[(diagnosis_df['Type'] == 'Normal')]
            inverted_data = diagnosis_df[(diagnosis_df['Type'] == 'Inverted') & (diagnosis_df['Inverted Metadata'] == attribute)]

            # Derreter o DataFrame para visualização de todas as classes
            original_melted = original_data.melt(id_vars=['Inverted Metadata', 'Type'], value_vars=['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK'],
                                                 var_name='Class', value_name='Probability')
            inverted_melted = inverted_data.melt(id_vars=['Inverted Metadata', 'Type'], value_vars=['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK'],
                                                 var_name='Class', value_name='Probability')

            # Plotar dados originais e invertidos
            sns.lineplot(data=original_melted, x='Class', y='Probability', label='Original', color='blue', marker='o')
            sns.lineplot(data=inverted_melted, x='Class', y='Probability', label='Inverted', color='orange', marker='o')

            plt.xlabel('Lesion Class')
            plt.ylabel('Probability (%)')
            plt.title(f'Impact of {attribute.capitalize()} on {diagnosis} Classification Probabilities')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()

        # Salvar figura
        plt.savefig(os.path.join(os.path.dirname(results_path), f'impact_{diagnosis}_improved.png'))
        plt.show()
else:
    # Carregar o modelo
    MODEL = set_and_load_model()

    # Carregar metadados
    metadata = pd.read_csv(metadata_path)

    # Filtrar para lesões de melanoma, BCC e SCC
    selected_metadata = metadata[metadata['diagnostic'].isin(['MEL', 'BCC', 'SCC'])]

    # Inicializar listas para armazenar resultados
    combined_results = []

    # Iterar sobre cada lesão selecionada
    for _, row in selected_metadata.iterrows():
        img_id = row['img_id']
        img_path = os.path.join(images_path, img_id)
        
        if not os.path.exists(img_path):
            continue  # Pular se o arquivo de imagem não existir
        
        img = Image.open(img_path).convert('RGB')  # Converter imagem para RGB
        
        # Converter metadados
        mapped_metadata = map_metadata(row)

        # Realizar a inferência original
        original_pred, _, _ = make_inference(MODEL, img, **mapped_metadata)
        combined_results.append({
            "Type": "Normal",
            "Inverted Metadata": "None",
            **{attr: row[attr] for attr in attributes},  # Incluindo apenas os atributos de interesse
            "diagnostic": row['diagnostic'],
            "ACK": original_pred[0],
            "BCC": original_pred[1],
            "MEL": original_pred[2],
            "NEV": original_pred[3],
            "SCC": original_pred[4],
            "SEK": original_pred[5]
        })
        
        # Iterar sobre cada atributo para inverter individualmente e analisar o impacto
        for attribute in attributes:
            # Manter todos os atributos fixos, exceto o que será invertido
            metadata_copy = row.copy()
            metadata_copy[attribute] = 'Sim' if row[attribute] == 'Não' else 'Não'
            
            # Converter metadados
            mapped_metadata_inverted = map_metadata(metadata_copy)

            # Realizar a inferência com o valor invertido
            inverted_pred, _, _ = make_inference(MODEL, img, **mapped_metadata_inverted)
            combined_results.append({
                "Type": "Inverted",
                "Inverted Metadata": attribute,
                **{attr: metadata_copy[attr] for attr in attributes},  # Incluindo apenas os atributos de interesse
                "diagnostic": row['diagnostic'],
                "ACK": inverted_pred[0],
                "BCC": inverted_pred[1],
                "MEL": inverted_pred[2],
                "NEV": inverted_pred[3],
                "SCC": inverted_pred[4],
                "SEK": inverted_pred[5]
            })

    # Converter resultados para DataFrame
    df = pd.DataFrame(combined_results)

    # Salvar DataFrame em CSV para referência futura
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.to_csv(results_path, index=False)

    # Melhor visualização para impactos individuais
    diagnostics = ['MEL', 'BCC', 'SCC']
    for diagnosis in diagnostics:
        plt.figure(figsize=(20, 20))

        diagnosis_df = df[df['diagnostic'] == diagnosis]

        # Iterar sobre cada atributo de metadados para criar subplots separados
        for i, attribute in enumerate(attributes, 1):
            plt.subplot(4, 3, i)

            # Extrair dados originais e invertidos para o atributo atual
            original_data = diagnosis_df[(diagnosis_df['Type'] == 'Normal')]
            inverted_data = diagnosis_df[(diagnosis_df['Type'] == 'Inverted') & (diagnosis_df['Inverted Metadata'] == attribute)]

            # Derreter o DataFrame para visualização de todas as classes
            original_melted = original_data.melt(id_vars=['Inverted Metadata', 'Type'], value_vars=['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK'],
                                                 var_name='Class', value_name='Probability')
            inverted_melted = inverted_data.melt(id_vars=['Inverted Metadata', 'Type'], value_vars=['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK'],
                                                 var_name='Class', value_name='Probability')

            # Plotar dados originais e invertidos
            sns.lineplot(data=original_melted, x='Class', y='Probability', label='Original', color='blue', marker='o', palette='bright')
            sns.lineplot(data=inverted_melted, x='Class', y='Probability', label='Inverted', color='blue', marker='o', palette='bright')

            plt.xlabel('Lesion Class')
            plt.ylabel('Probability (%)')
            plt.title(f'Impact of {attribute.capitalize()} on {diagnosis} Classification Probabilities')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()

        # Salvar figura
        plt.savefig(os.path.join(os.path.dirname(results_path), f'impact_{diagnosis}_improved.png'))
        plt.show()
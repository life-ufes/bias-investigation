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

# Fazer o DataFrame para plotagem para cada diagnóstico
for diagnosis in ['MEL', 'BCC', 'SCC']:
    diagnosis_df = df[df['Diagnosis'] == diagnosis]
    melted_df = diagnosis_df.melt(id_vars='Age', value_vars=['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK'],
                                  var_name='Class', value_name='Probability')

    # Plotar os dados usando o gráfico de linha do Seaborn
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=melted_df, x='Age', y='Probability', hue='Class', ci='sd')
    plt.xlabel('Age')
    plt.ylabel('Probability (%)')
    plt.title(f'Impact of Age on {diagnosis} Classification Probabilities with Error Bands')
    plt.grid(True)
    plt.show()

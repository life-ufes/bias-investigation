import sys
import os
import json

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

# Importe as bibliotecas do projeto
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.model_inference import set_and_load_model, make_inference
from utils.data_loader import load_metadata, load_image
from utils.metadata_mapper import map_metadata

# Paths (agora obtidos do arquivo de configuração)
metadata_path = os.path.join(dataset_path, 'metadata.csv')
images_path = os.path.join(dataset_path, 'images')

# Load the model
MODEL = set_and_load_model()

# Load metadata
metadata = load_metadata(metadata_path)

# Filter for melanoma lesions
melanoma_metadata = metadata[metadata['diagnostic'] == 'MEL']

# Select a random melanoma lesion
random_row = melanoma_metadata.sample(n=1).iloc[0]

# Load image
try:
    img = load_image(images_path, random_row['img_id'])
except FileNotFoundError as e:
    print(e)
    exit()

# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title(f"Melanoma Image: {random_row['img_id']}")
plt.axis('off')
plt.show()

# Map metadata
mapped_metadata = map_metadata(random_row)

# Make inference
pred, pred_label, pred_prob = make_inference(
    MODEL, img,
    mapped_metadata['region'], mapped_metadata['fitzpatrick'], mapped_metadata['age'],
    mapped_metadata['gender'], mapped_metadata['smoke'], mapped_metadata['drink'],
    mapped_metadata['itch'], mapped_metadata['grew'], mapped_metadata['bleed'],
    mapped_metadata['hurt'], mapped_metadata['changed'], mapped_metadata['elevation'],
    mapped_metadata['cancer_history'], mapped_metadata['skin_cancer_history'], verbose=False
)

# Prepare results for display
results = {
    "Classe da Lesão": ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"],
    "Probabilidade (%)": [f"{p * 100:.2f}%" for p in pred]
}

metadata_results = {
    "Metadado": ["Gender", "Smoke", "Drink", "Itch", "Grew", "Bleed", "Hurt", "Changed", "Elevation", "Cancer History", "Skin Cancer History", "Age", "Region", "Fitzpatrick"],
    "Valor": [mapped_metadata['gender'], mapped_metadata['smoke'], mapped_metadata['drink'], mapped_metadata['itch'],
              mapped_metadata['grew'], mapped_metadata['bleed'], mapped_metadata['hurt'], mapped_metadata['changed'],
              mapped_metadata['elevation'], mapped_metadata['cancer_history'], mapped_metadata['skin_cancer_history'],
              mapped_metadata['age'], mapped_metadata['region'], mapped_metadata['fitzpatrick']]
}

results_df = pd.DataFrame(results)
metadata_df = pd.DataFrame(metadata_results)

# Display metadata and results in a table format using matplotlib
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].axis('tight')
ax[0].axis('off')
ax[0].table(cellText=metadata_df.values, colLabels=metadata_df.columns, cellLoc='center', loc='center')

ax[1].axis('tight')
ax[1].axis('off')
ax[1].table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')

plt.show()

# Experiment 2 - Grad-CAM Images and Individual Metadata Impact Visualization

import os
import sys
import json
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.transforms import transforms
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Root directory and paths
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(root_dir, 'config.json')

# Load configuration
try:
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print(f"Error: Configuration file not found at {config_path}")
    sys.exit(1)

dataset_path = config.get('dataset_path', './PAD-UFES-20')
metadata_path = os.path.join(dataset_path, 'metadata.csv')
images_path = os.path.join(dataset_path, 'images')

# Load utilities
sys.path.insert(0, os.path.join(root_dir, 'utils'))
from model_inference import set_and_load_model, make_inference
from metadata_mapper import map_metadata

# Attributes to analyze
attributes = ['bleed', 'cancer_history', 'changed', 'drink', 'elevation', 
              'grew', 'hurt', 'itch', 'skin_cancer_history', 'smoke']

# Output directories
output_dir = os.path.join(root_dir, 'results', 'gradcam_images', 'experiment_2')
impact_dir = os.path.join(root_dir, 'results', 'individual_metadata_impact')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(impact_dir, exist_ok=True)

# Function to normalize Grad-CAM heatmap
def normalize_cam(cam):
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize to [0, 1]
    return cam

# Function to prepare heatmap
def prepare_heatmap(heatmap, img_shape):
    heatmap = normalize_cam(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img_shape[1], img_shape[0]))
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = np.repeat(heatmap, 3, axis=-1)
    return heatmap

# Load model
model = set_and_load_model()

# Identify the last convolutional layer
target_layer = model.features[-2][-1]

# Load metadata
metadata = pd.read_csv(metadata_path)
selected_metadata = metadata[metadata['diagnostic'].isin(['MEL', 'BCC', 'SCC'])]

# Initialize results storage
combined_results = []

for _, row in selected_metadata.iterrows():
    img_id = row['img_id']
    diagnosis = row['diagnostic']
    img_path = os.path.join(images_path, img_id)
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert('RGB')
    mapped_metadata = map_metadata(row)

    # Diagnosis and image ID directories
    diagnosis_dir = os.path.join(output_dir, diagnosis)
    img_id_dir = os.path.join(diagnosis_dir, img_id)
    os.makedirs(img_id_dir, exist_ok=True)

    for attribute in attributes:
        # Generate Grad-CAM for Normal metadata
        pred, pred_label, pred_prob = make_inference(
            model, img, mapped_metadata['region'], mapped_metadata['fitzpatrick'], mapped_metadata['age'],
            mapped_metadata['gender'], mapped_metadata['smoke'], mapped_metadata['drink'], mapped_metadata['itch'],
            mapped_metadata['grew'], mapped_metadata['bleed'], mapped_metadata['hurt'], mapped_metadata['changed'],
            mapped_metadata['elevation'], mapped_metadata['cancer_history'], mapped_metadata['skin_cancer_history'],
            verbose=False
        )
        combined_results.append({
            "lesion_id": img_id,
            "Type": "Normal",
            "Inverted Metadata": "None",
            "Attribute": attribute,
            "diagnostic": diagnosis,
            **{attr: row[attr] for attr in attributes},
            "ACK": pred[0],
            "BCC": pred[1],
            "MEL": pred[2],
            "NEV": pred[3],
            "SCC": pred[4],
            "SEK": pred[5]
        })

        # Generate Grad-CAM for Inverted metadata
        inverted_metadata = mapped_metadata.copy()
        inverted_metadata[attribute] = 'Sim' if mapped_metadata[attribute] == 'Não' else 'Não'
        pred, pred_label, pred_prob = make_inference(
            model, img, inverted_metadata['region'], inverted_metadata['fitzpatrick'], inverted_metadata['age'],
            inverted_metadata['gender'], inverted_metadata['smoke'], inverted_metadata['drink'], inverted_metadata['itch'],
            inverted_metadata['grew'], inverted_metadata['bleed'], inverted_metadata['hurt'], inverted_metadata['changed'],
            inverted_metadata['elevation'], inverted_metadata['cancer_history'], inverted_metadata['skin_cancer_history'],
            verbose=False
        )
        combined_results.append({
            "lesion_id": img_id,
            "Type": "Inverted",
            "Inverted Metadata": attribute,
            "Attribute": attribute,
            "diagnostic": diagnosis,
            **{attr: inverted_metadata[attr] for attr in attributes},
            "ACK": pred[0],
            "BCC": pred[1],
            "MEL": pred[2],
            "NEV": pred[3],
            "SCC": pred[4],
            "SEK": pred[5]
        })

# Save combined results to CSV
results_df = pd.DataFrame(combined_results)
results_csv_path = os.path.join(root_dir, 'results', 'individual_metadata_impact.csv')
results_df.to_csv(results_csv_path, index=False)

# Generate individual impact graphs
for attribute in attributes:
    for diagnosis in ['MEL', 'BCC', 'SCC']:
        plt.figure(figsize=(10, 6))
        diagnosis_data = results_df[(results_df['diagnostic'] == diagnosis) & 
                                    (results_df['Inverted Metadata'] == attribute)]
        
        # Melt data for visualization
        melted = diagnosis_data.melt(id_vars=['Type'], value_vars=['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK'],
                                      var_name='Class', value_name='Probability')

        # Plot data
        sns.lineplot(data=melted[melted['Type'] == 'Normal'], x='Class', y='Probability', label='Normal', marker='o')
        sns.lineplot(data=melted[melted['Type'] == 'Inverted'], x='Class', y='Probability', label='Inverted', marker='o')
        plt.title(f"Impact of {attribute.capitalize()} on {diagnosis}")
        plt.xlabel("Lesion Class")
        plt.ylabel("Probability (%)")
        plt.xticks(rotation=45)
        plt.grid(True)

        # Save individual plot
        output_plot_path = os.path.join(impact_dir, f"{diagnosis}_{attribute}_impact.png")
        plt.savefig(output_plot_path)
        plt.close()
        print(f"Saved plot: {output_plot_path}")

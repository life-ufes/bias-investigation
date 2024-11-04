import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load results from CSV
results_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'results', 'individual_metadata_impact.csv')
df = pd.read_csv(results_path)

# Ensure the 'diagnostic' column is available
diagnostics = ['MEL', 'BCC', 'SCC']
if 'diagnostic' not in df.columns:
    print("Error: 'diagnostic' column not found in the DataFrame.")
    sys.exit(1)

# Improved visualization for individual impacts
for diagnosis in diagnostics:
    plt.figure(figsize=(20, 20))

    diagnosis_df = df[df['diagnostic'] == diagnosis]

    # Iterate over each metadata attribute to create separate subplots
    for i, attribute in enumerate(attributes, 1):
        plt.subplot(4, 3, i)

        # Extract original and inverted data for the current attribute
        original_data = diagnosis_df[(diagnosis_df['Type'] == 'Normal') & (diagnosis_df['Inverted Metadata'] == 'None')]
        inverted_data = diagnosis_df[(diagnosis_df['Type'] == 'Inverted') & (diagnosis_df['Inverted Metadata'] == attribute)]

        # Melt the DataFrame for visualization of all classes
        original_melted = original_data.melt(id_vars=['Inverted Metadata', 'Type'], value_vars=['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK'],
                                             var_name='Class', value_name='Probability')
        inverted_melted = inverted_data.melt(id_vars=['Inverted Metadata', 'Type'], value_vars=['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK'],
                                             var_name='Class', value_name='Probability')

        # Plot original and inverted data
        sns.lineplot(data=original_melted, x='Class', y='Probability', label='Original', color='blue', marker='o')
        sns.lineplot(data=inverted_melted, x='Class', y='Probability', label='Inverted', color='orange', marker='o')

        plt.xlabel('Lesion Class')
        plt.ylabel('Probability (%)')
        plt.title(f'Impact of {attribute.capitalize()} on {diagnosis} Classification Probabilities')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(os.path.dirname(results_path), f'impact_{diagnosis}_improved.png'))
    plt.show()

import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# Configuration
csv_path = "results/individual_metadata_impact.csv"
output_dir = "results/metadata_impact_analysis_v2"
os.makedirs(output_dir, exist_ok=True)

# Parameters
lesion_classes = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
metadata_columns = ['bleed', 'cancer_history', 'changed', 'elevation', 'grew', 'hurt', 'itch', 'Inverted Metadata']
diagnostics = ['BCC', 'MEL', 'SCC']
alpha = 0.05

# Load data
df = pd.read_csv(csv_path).query("diagnostic in @diagnostics")

# Helper function
def calculate_effects(normal, inverted, target_class):
    """Calculate clinical effect sizes and statistical significance"""
    # Convert to numeric arrays
    normal_vals = pd.to_numeric(normal, errors='coerce')
    inverted_vals = pd.to_numeric(inverted, errors='coerce')
    
    effects = {
        'abs_change': np.abs(inverted_vals - normal_vals).mean(),
        'rel_change': 0,
        'pval': 1.0
    }
    
    baseline_risk = normal_vals.mean()
    if baseline_risk > 0:
        effects['rel_change'] = effects['abs_change'] / baseline_risk
        
    try:
        _, effects['pval'] = wilcoxon(normal_vals, inverted_vals)
    except ValueError:
        pass
        
    return effects

# Analysis pipeline
results = []
for diagnosis in diagnostics:
    print(f"\nAnalyzing {diagnosis}")
    
    # Process data
    diag_df = (
        df[df['diagnostic'] == diagnosis]
        .groupby(['lesion_id', 'Type'])
        .agg({**{col: 'first' for col in metadata_columns},
            **{col: 'mean' for col in lesion_classes}})
        .reset_index()
    )
    diag_df[lesion_classes] = diag_df[lesion_classes].apply(pd.to_numeric, errors='coerce')

    # Pivot table
    paired = diag_df.pivot(
        index='lesion_id',
        columns='Type',
        values=metadata_columns + lesion_classes
    )
    
    # Analyze attributes
    attr_results = []
    for attr in metadata_columns[:-1]:  # Exclude 'Inverted Metadata' itself
        mask = paired[('Inverted Metadata', 'Inverted')] == attr
        valid_pairs = paired[mask]
        
        if not valid_pairs.empty:
            normal_probs = pd.to_numeric(
                valid_pairs.xs('Normal', axis=1, level=1)[diagnosis], 
                errors='coerce'
            )
            inverted_probs = pd.to_numeric(
                valid_pairs.xs('Inverted', axis=1, level=1)[diagnosis],
                errors='coerce'
            )            
            effects = calculate_effects(normal_probs, inverted_probs, diagnosis)
            attr_results.append({
                'Attribute': attr,
                'N': len(valid_pairs),
                **effects
            })
    
    # Process results
    impact_df = pd.DataFrame(attr_results)
    impact_df['qval'] = multipletests(impact_df['pval'], method='fdr_bh')[1]
    impact_df['significant'] = impact_df['qval'] < alpha
    impact_df = impact_df.sort_values('abs_change', ascending=False)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=impact_df,
        x='abs_change',
        y='Attribute',
        hue='significant',
        palette={True: '#e41a1c', False: '#377eb8'},
        dodge=False
    )
    
    plt.title(f"Metadata Impact on {diagnosis} Diagnosis")
    plt.xlabel("Absolute Probability Change")
    plt.ylabel(None)
    plt.axvline(0.05, ls='--', c='gray')
    
    for i, row in impact_df.iterrows():
        plt.text(
            row['abs_change'] + 0.01,
            i,
            f"Δ={row['abs_change']:.2f}\n(p={row['qval']:.3f}",
            va='center',
            fontsize=9
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{diagnosis}_impact.png"))
    plt.close()
    
    results.append(impact_df.sort_values('abs_change', ascending=False))

# Save summary
pd.concat(results).to_csv(os.path.join(output_dir, "metadata_impact_summary.csv"), index=False)
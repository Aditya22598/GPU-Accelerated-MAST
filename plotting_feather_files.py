#!/usr/bin/env python3
import shutil
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def compare(cell_type: str):
    # Paths to Feather files
    change_path = f"change_output/{cell_type}.feather"
    vanilla_path = f"output_ultrafast/{cell_type}.feather"

    # Load data
    try:
        df_change = pd.read_feather(change_path)
        df_vanilla = pd.read_feather(vanilla_path)
    except ImportError:
        sys.exit("Error: pyarrow is required to read Feather files. Install it via 'pip install pyarrow'.")

    # Rename columns for alignment
    df_change = df_change.rename(columns={'index': 'gene_symbol', 'proba_m1': 'posterior_prob_DE'})
    df_vanilla = df_vanilla.rename(columns={'index': 'gene_symbol', 'proba_m1': 'posterior_prob_DE'})

    # Clean gene_symbol: uppercase, stripped, string
    for df in [df_change, df_vanilla]:
        df['gene_symbol'] = df['gene_symbol'].astype(str).str.strip().str.upper()

    # Select relevant columns and drop duplicates, sort by gene_symbol
    df_change_sorted = df_change[['gene_symbol', 'posterior_prob_DE']].drop_duplicates('gene_symbol').sort_values('gene_symbol').reset_index(drop=True)
    df_vanilla_sorted = df_vanilla[['gene_symbol', 'posterior_prob_DE']].drop_duplicates('gene_symbol').sort_values('gene_symbol').reset_index(drop=True)

    # Merge: vanilla left for order, change on right
    df_merged = pd.merge(
        df_vanilla_sorted,
        df_change_sorted,
        on='gene_symbol',
        how='inner',
        suffixes=('_vanilla', '_change')
    ).reset_index(drop=True)

    # Print debugging info
    print(f"Total unique gene symbols - vanilla: {len(df_vanilla_sorted)}, change: {len(df_change_sorted)}, merged: {len(df_merged)}")
    print("First few merged rows:")
    print(df_merged.head(5))

    # --------- SAVE FIGURES IN "figures" FOLDER ---------

    chunk_size = 50
    num_chunks = (len(df_merged) + chunk_size - 1) // chunk_size
    os.makedirs(f'figures/{cell_type}', exist_ok=True)
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df_merged))
        chunk = df_merged.iloc[start_idx:end_idx]
        x = chunk['gene_symbol']

        plt.figure(figsize=(max(12, 0.25 * len(chunk)), 6))
        plt.plot(x, chunk['posterior_prob_DE_vanilla'], label='vanilla', marker='x')
        plt.plot(x, chunk['posterior_prob_DE_change'], label='change', marker='o')
        plt.xlabel('Gene Symbol')
        plt.ylabel('Posterior Probability')
        plt.title(f'Posterior Probabilities: Genes {start_idx+1}-{end_idx}')
        plt.xticks(rotation=90, fontsize=7)
        plt.legend()
        plt.tight_layout()
        filename = f'posterior_prob_comp_{start_idx+1:03}_{end_idx:03}.png'
        plt.savefig(f"figures/{cell_type}/{filename}", dpi=200)
        plt.close()
    print(f"\nSaved {num_chunks} comparison plots (50 genes per plot) in the ./figures/{cell_type} directory.")


if __name__ == "__main__":
    if os.path.exists("figures"):
        shutil.rmtree("figures")
    os.makedirs("figures", exist_ok=True)
    for root, dirs, files in os.walk("change_output"):
        for file in files:
            cell_names: str = file[:-len('.feather')]
            #compare(cell_names)

    print("Done.")
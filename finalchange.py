#!/usr/bin/env python3
import os
import gc
import json
import shutil
from datetime import datetime

import numpy as np
import scanpy as sc
import scvi
import torch
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
MAX_EPOCHS = 60
BATCH_SIZE = 2048
LATENT_DIM = 5
HIDDEN_UNITS = 64
HIDDEN_LAYERS = 1
DROPOUT_RATE = 0.0
USE_AMPMIXED16 = True
CONVERT_TO_DENSE = True
NUM_DL_WORKERS = min(4, os.cpu_count() or 1)
MIN_CELLS_PER_GENE_FRAC = 0.10

DE_N_SAMPLES = 200
DE_ALL_STATS = False
MC_NORM_SAMPLES = 100

# ============================================================================
# Hardware Configuration
# ============================================================================
GPU_AVAILABLE = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
scvi.settings.dl_num_workers = NUM_DL_WORKERS
scvi.settings.dl_persistent_workers = True
scvi.settings.num_threads = NUM_DL_WORKERS

print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Running on {'GPU' if GPU_AVAILABLE else 'CPU'}, AMP16={USE_AMPMIXED16}")

# ============================================================================
# Paths
# ============================================================================
MASK = "cell_type"
ANN_PATH_ALL = "/20TB-storage/aditya22598/zoo/adata_raw.h5ad"
CELL_JSON = "/20TB-storage/aditya22598/zoo/cell_type_data.json"
OUTPUT_DIR = "/20TB-storage/aditya22598/zoo/change_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Load AnnData
# ============================================================================
print(f"[{datetime.now():%H:%M:%S}] Loading AnnData...")
adata_all = sc.read_h5ad(ANN_PATH_ALL)
adata_all.var_names_make_unique()
print(f"Loaded {adata_all.n_obs:,} cells × {adata_all.n_vars:,} genes")

if CONVERT_TO_DENSE and not isinstance(adata_all.X, np.ndarray):
    print("Converting to dense representation...")
    adata_all.X = adata_all.X.toarray()

# ============================================================================
# Run DE per cell type
# ============================================================================
def run_cell_type(ct: str):
    try:
        print(f"\n[{datetime.now():%H:%M:%S}] Processing: {ct}")
        out_dir = os.path.join(OUTPUT_DIR, ct)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)

        adata = adata_all[adata_all.obs[MASK] == ct].copy()
        if adata.n_obs < 20:
            print(f"Skipped {ct}: Too few cells ({adata.n_obs})")
            return

        sc.pp.filter_genes(
            adata,
            min_cells=int(MIN_CELLS_PER_GENE_FRAC * adata.n_obs)
        )
        adata.obs["ngeneson"] = (
            adata.obs["n_genes_by_counts"] - adata.obs["n_genes_by_counts"].mean()
        ) / adata.obs["n_genes_by_counts"].std()

        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key="replicate" if "replicate" in adata.obs.columns else None,
            labels_key="label",
            continuous_covariate_keys=["ngeneson"]
        )

        model = scvi.model.SCVI(
            adata,
            n_latent=LATENT_DIM,
            n_hidden=HIDDEN_UNITS,
            n_layers=HIDDEN_LAYERS,
            dropout_rate=DROPOUT_RATE
        )

        model.train(
            max_epochs=MAX_EPOCHS,
            early_stopping=True,
            batch_size=BATCH_SIZE,
            accelerator="gpu" if GPU_AVAILABLE else "cpu",
            devices=1 if GPU_AVAILABLE else None,
            precision="16-mixed" if USE_AMPMIXED16 and GPU_AVAILABLE else "32",
            load_sparse_tensor=not CONVERT_TO_DENSE,
        )

        if GPU_AVAILABLE:
            model.to_device("cuda:0")

        norm = model.get_normalized_expression(
            n_samples=MC_NORM_SAMPLES,
            return_mean=True
        )
        adata.obsm["normalized"] = norm

        # --- KEY CHANGE: Use "change" mode and "importance" weights ---
        de = model.differential_expression(
            idx1=(adata.obs["label"] == "ctrl"),
            idx2=(adata.obs["label"] == "stim"),
            mode="change",                  # <--- Use "change" mode
            batch_correction=True,
            weights="importance",           # <--- Use "importance" weights
            n_samples_overall=DE_N_SAMPLES,
            all_stats=DE_ALL_STATS
        )

        df = (
            de.reset_index()
            .rename(columns={
                "index": "gene_symbol",
                "proba_de": "posterior_prob_DE",
                "lfc_mean": "log2_fold_change",
                "bayes_factor": "bayes_factor"
            })
            .assign(cell_type=ct)
        )
        fea_path = os.path.join(out_dir, f"{ct}_de_ultrafast.feather")
        df.to_feather(fea_path)
        print(f"Saved results for {ct}: {df.shape[0]:,} genes")

    except Exception as e:
        print(f"Error while processing {ct}: {e}")
    finally:
        # Clean up memory and cache
        del adata, model, de, norm, df
        gc.collect()
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()

# ============================================================================
# Main Execution
# ============================================================================
with open(CELL_JSON) as f:
    cell_types = json.load(f)

print(f"Starting DE processing for {len(cell_types)} cell types...")
for ct in tqdm(cell_types, ncols=80):
    run_cell_type(ct)

print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Pipeline completed. Results saved in: {OUTPUT_DIR}")


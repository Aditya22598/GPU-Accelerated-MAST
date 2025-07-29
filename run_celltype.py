import os
import sys
import gc
import numpy as np
import scanpy as sc
import scvi
import torch
from datetime import datetime

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
MIN_CELLS_PER_GENE_FRAC = 0.10
DE_N_SAMPLES = 200
DE_ALL_STATS = False
MC_NORM_SAMPLES = 100
ITR = 300  # Number of label shuffling iterations

# ============================================================================
# Hardware Configuration
# ============================================================================
GPU_AVAILABLE = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
scvi.settings.dl_num_workers = 0
scvi.settings.dl_persistent_workers = False
scvi.settings.num_threads = 1

MASK = "cell_type"
ANN_PATH_ALL = "/20TB-storage/aditya22598/zoo/adata_raw.h5ad"
OUTPUT_DIR = "/20TB-storage/aditya22598/zoo/output_ultrafast_null"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Running on {'GPU' if GPU_AVAILABLE else 'CPU'}, AMP16={USE_AMPMIXED16}")

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
# Run DE for one iteration of one cell type (with shuffled labels)
# ============================================================================
def run_cell_type_iteration(ct: str, iteration: int):
    try:
        print(f"\n[{datetime.now():%H:%M:%S}] Cell type: {ct} | Iteration: {iteration}")
        out_dir = os.path.join(OUTPUT_DIR, ct)
        os.makedirs(out_dir, exist_ok=True)
        
        adata = adata_all[adata_all.obs[MASK] == ct].copy()
        
        if adata.n_obs < 20:
            print(f"Skipped {ct}: Too few cells ({adata.n_obs})")
            return
        
        sc.pp.filter_genes(adata, min_cells=int(MIN_CELLS_PER_GENE_FRAC * adata.n_obs))
        
        adata.obs["ngeneson"] = (
            adata.obs["n_genes_by_counts"] - adata.obs["n_genes_by_counts"].mean()
        ) / adata.obs["n_genes_by_counts"].std()
        
        # Shuffle the labels
        adata.obs["label"] = np.random.permutation(adata.obs["label"].values)
        
        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key="replicate",
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
        
        de = model.differential_expression(
            idx1=(adata.obs["label"] == "ctrl"),
            idx2=(adata.obs["label"] == "stim"),
            mode="vanilla",
            batch_correction=True,
            weights="uniform",
            n_samples_overall=DE_N_SAMPLES,
            all_stats=DE_ALL_STATS
        )
        
        df = (
            de.reset_index()
            .rename(columns={
                "index": "gene_symbol",
                "proba_m1": "Pr(ctrl>stim)",
                "bayes_factor": "bayes_factor"
            })
            .assign(
                cell_type=ct,
                iteration=iteration,
                log2_fold_change=lambda x: np.log2((x.scale1 + 1e-9) / (x.scale2 + 1e-9))
            )
        )
        
        fea_path = os.path.join(out_dir, f"Itr_{iteration:03d}.feather")
        df.to_feather(fea_path)
        print(f"Saved iteration {iteration} results for {ct}: {df.shape[0]:,} genes")
        
    except Exception as e:
        print(f"Error while processing {ct} (Itr {iteration}): {e}")
    finally:
        # Clean up memory
        for _v in ("adata", "model", "norm", "de"):
            if _v in locals():
                del locals()[_v]
        gc.collect()
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()

# ============================================================================
# Run all permutations for one cell type
# ============================================================================
def run_permutations_for_cell_type(ct: str):
    for i in range(1, ITR + 1):
        run_cell_type_iteration(ct, i)
    # Clear SCVI model registry and memory after finishing the cell type
    if hasattr(scvi.model.base, '_MODEL_REGISTRY'):
        scvi.model.base._MODEL_REGISTRY.clear()
    gc.collect()
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()
    print(f"Completed {ct}, memory cleaned")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_celltype.py <cell_type>")
        sys.exit(1)
    cell_type = sys.argv[1]
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Starting DE null-distribution estimation for cell type: {cell_type}")
    run_permutations_for_cell_type(cell_type)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Finished cell type: {cell_type}")


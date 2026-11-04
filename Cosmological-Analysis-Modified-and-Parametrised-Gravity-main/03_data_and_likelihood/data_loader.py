# data_loader.py
import os
import numpy as np
from scipy.linalg import cho_factor, cho_solve

# Inside data_loader.py
import os

# 1. Locate where data_loader.py itself lives on your machine
# This will be: .../emcee_project/Thesis_Project_data_analysis/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build paths directly relative to data_loader.py's location
# Since 'datasets' is in the same folder as data_loader.py, target it directly:
PANTHEON_PATH = os.path.join(SCRIPT_DIR, "datasets", "SH0ES_datasets", "Pantheon+SH0ES.dat")
COV_MATRIX_PATH = os.path.join(SCRIPT_DIR, "datasets", "SH0ES_datasets", "sys_full_long.txt")

C_LIGHT = 299792.458
DATA_DIR = SCRIPT_DIR
CC_FILE  = os.path.join(DATA_DIR , "datasets/cc_data.txt")
BAO_FILE = os.path.join(DATA_DIR ,"datasets/bao_data.txt")
CLU_FILE = os.path.join(DATA_DIR ,"datasets/clustering_data.txt")
BAO_KEEP_INDICES = [0, 1, 2, 3, 4, 5]  # Deduplication

def load_pantheon_data():
    dat_path = os.path.join(DATA_DIR, "datasets/SH0ES_datasets/Pantheon+SH0ES.dat")
    cov_path = os.path.join(DATA_DIR, "datasets/SH0ES_datasets/Pantheon+SH0ES_STAT+SYS.cov")

    with open(dat_path) as f:
        header = f.readline().strip().lstrip('#').split()
    raw = np.genfromtxt(dat_path, names=header, skip_header=1)
    z_cmb = raw['zHD'].astype(float)
    mu_obs = raw['MU_SH0ES'].astype(float)

    with open(cov_path) as f:
        N_cov = int(f.readline().strip())
        cov_flat = np.fromstring(f.read(), sep=' ')

    C = cov_flat.reshape(N_cov, N_cov)
    C_fac = cho_factor(C, lower=True, check_finite=False)
    
    # Precompute vectors
    ones = np.ones(len(z_cmb))
    Cinv_ones = cho_solve(C_fac, ones, check_finite=False)
    scalar_denom = float(ones @ Cinv_ones)
    
    return z_cmb, mu_obs, C, C_fac, Cinv_ones, scalar_denom

def _load_hz_file(filepath, keep_indices=None):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    data = np.loadtxt(filepath, comments='#')
    if keep_indices is not None:
        data = data[keep_indices]
    return data[:, 0], data[:, 1], data[:, 2]

def load_cc_data(): return _load_hz_file(CC_FILE)
def load_bao_data(): return _load_hz_file(BAO_FILE, keep_indices=BAO_KEEP_INDICES)
def load_clustering_data(): return _load_hz_file(CLU_FILE)
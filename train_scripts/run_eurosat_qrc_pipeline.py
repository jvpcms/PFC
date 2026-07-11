"""End-to-end EuroSAT QRC denoising pipeline (standalone).

Replicates Denoise_EuroSAT_QRC.ipynb:
  1. Download/extract EuroSAT RGB, grayscale, 80/10/10 stratified split
  2. Multiplicative speckle noise (sigma)
  3. PCA(100) fit on clean train, project noisy splits
  4. First d components -> min-max scale -> Rydberg reservoir embeddings (GPU, cached)
  5. Standardise embeddings, train hybrid readout MLP (subprocess)
  6. Train classical PCA baseline MLP (subprocess)

Run from the repo root:
  python train_scripts/run_eurosat_qrc_pipeline.py --sigma 0.7 --d-qrc 18

Outputs (same layout as the notebook):
  models/eurosat_qrc/sigma{S}_embeddings_d{D}/{train,val,test}.npy
  models/eurosat_qrc/sigma{S}_embed_scaler_d{D}.joblib
  models/eurosat_qrc/sigma{S}_readout_d{D}.keras
  models/eurosat_qrc/sigma{S}_baseline_pca{D}.keras
"""

import argparse
import os
import random
import sys
import subprocess
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

SEED = 42
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"

DATA_DIR = Path("data/EuroSAT_RGB")
ZIP_PATH = Path("data_zip/EuroSAT_RGB.zip")
MODEL_DIR = Path("models/eurosat_qrc")

N_COMPONENTS = 100
BATCH_SIZE_QRC = 32
CHEB_TOL = 1e-7

# MLP training hyperparameters (identical for baseline and hybrid)
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.3
BATCH_SIZE = 64
PATIENCE = 20
EPOCHS = 500


# ----------------------------------------------------------------------
# 1. Dataset
# ----------------------------------------------------------------------

def to_grayscale(X_rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luminance. Input: (..., 3) float32. Output: (...) float32."""
    return 0.299 * X_rgb[..., 0] + 0.587 * X_rgb[..., 1] + 0.114 * X_rgb[..., 2]


def download_eurosat():
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        return
    if not ZIP_PATH.exists():
        ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading EuroSAT RGB from {EUROSAT_URL} ...")
        urllib.request.urlretrieve(EUROSAT_URL, ZIP_PATH)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR.parent)


def load_dataset():
    from PIL import Image

    categories = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
    label_map = {cat: i for i, cat in enumerate(categories)}
    print("Categories:", categories)

    images, labels = [], []
    for cat in categories:
        for path in sorted((DATA_DIR / cat).glob("*.jpg")):
            img = np.array(Image.open(path).resize((64, 64)), dtype=np.float32) / 255.0
            images.append(img)
            labels.append(label_map[cat])

    X = to_grayscale(np.stack(images))  # (N, 64, 64)
    y = np.array(labels)
    print(f"Total: {X.shape}")
    return X, y


def split_dataset(X, y):
    from sklearn.model_selection import train_test_split

    # 80% train, 10% val, 10% test — same seeds/stratification as the notebook
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test


# ----------------------------------------------------------------------
# 2. Noise
# ----------------------------------------------------------------------

def apply_noise(X: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """I~ = clip(I * (1 + sigma * N(0,1)), 0, 1). X must be float32 in [0,1]."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=X.shape).astype(np.float32)
    return np.clip(X * (1 + eps), 0.0, 1.0)


# ----------------------------------------------------------------------
# 4. Rydberg reservoir (CPU reference + GPU Chebyshev propagation)
# ----------------------------------------------------------------------

@dataclass
class QRCConfig:
    n_qubits: int = 18
    n_steps: int = 8
    dt: float = 0.5
    omega: float = 2 * np.pi       # Rabi frequency (rad/us)
    global_detuning: float = 4.5   # Delta_0 (rad/us)
    detuning_scale: float = 9.0    # k (rad/us)
    interaction_strength: float = 5.42  # C6/a^6, a = 10 um spacing (rad/us)
    interaction_power: float = 6.0
    shots: int | None = None       # None = exact expectation values
    seed: int = SEED


class RydbergReservoir:
    """1D Rydberg-atom chain quantum reservoir (CPU reference).

    H_fixed built once; per-image cost is a diagonal update + n_steps expm_multiply calls.
    """

    def __init__(self, config: QRCConfig):
        from scipy.sparse import csr_matrix, diags

        self.cfg = config
        n, dim = config.n_qubits, 2 ** config.n_qubits
        self.dim = dim
        self.rng = np.random.default_rng(config.seed)

        self._Z_eigs = np.stack([self._z_eig(i, n, dim) for i in range(n)])  # (n, dim)
        self._H_fixed = self._build_H_fixed()

        self._obs = list(self._Z_eigs)
        for i in range(n):
            for j in range(i + 1, n):
                self._obs.append(self._Z_eigs[i] * self._Z_eigs[j])

        n_obs = len(self._obs)
        self.embed_dim = config.n_steps * n_obs
        print(
            f"RydbergReservoir: d={n}, dim={dim}, "
            f"obs={n_obs} ({n} Z + {n*(n-1)//2} ZZ), embed_dim={self.embed_dim}"
        )

    @staticmethod
    def _z_eig(qubit: int, n_qubits: int, dim: int) -> np.ndarray:
        mask = 1 << (n_qubits - 1 - qubit)
        idx = np.arange(dim)
        bit = (idx & mask) >> (n_qubits - 1 - qubit)
        return (1 - 2 * bit).astype(np.float64)

    @staticmethod
    def _x_sparse(qubit: int, n_qubits: int, dim: int):
        from scipy.sparse import csr_matrix

        mask = 1 << (n_qubits - 1 - qubit)
        row = np.arange(dim)
        col = row ^ mask
        return csr_matrix(
            (np.ones(dim, dtype=np.complex128), (row, col)), shape=(dim, dim)
        )

    def _build_H_fixed(self):
        from scipy.sparse import csr_matrix, diags

        cfg = self.cfg
        n, dim = cfg.n_qubits, self.dim

        H = csr_matrix((dim, dim), dtype=np.complex128)
        for i in range(n):
            H = H + cfg.omega * self._x_sparse(i, n, dim)

        diag = np.zeros(dim, dtype=np.float64)
        for i in range(n):
            diag += 0.5 * cfg.global_detuning * self._Z_eigs[i]
        for i in range(n):
            for j in range(i + 1, n):
                vij = cfg.interaction_strength / (abs(j - i) ** cfg.interaction_power)
                diag += 0.25 * vij * self._Z_eigs[i]
                diag += 0.25 * vij * self._Z_eigs[j]
                diag += 0.25 * vij * self._Z_eigs[i] * self._Z_eigs[j]

        return H + diags(diag.astype(np.complex128))

    def _data_diag(self, z: np.ndarray) -> np.ndarray:
        return (-0.5 * self.cfg.detuning_scale) * (z @ self._Z_eigs)

    def transform_one(self, z: np.ndarray) -> np.ndarray:
        from scipy.sparse import diags
        from scipy.sparse.linalg import expm_multiply

        H = self._H_fixed + diags(self._data_diag(z).astype(np.complex128))
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0

        features = []
        for _ in range(self.cfg.n_steps):
            state = expm_multiply(-1j * H * self.cfg.dt, state)
            probs = np.abs(state) ** 2
            probs /= probs.sum()
            if self.cfg.shots is None:
                features.extend(float(np.dot(probs, obs)) for obs in self._obs)
            else:
                samples = self.rng.choice(self.dim, size=self.cfg.shots, p=probs)
                features.extend(float(obs[samples].mean()) for obs in self._obs)

        return np.array(features, dtype=np.float64)


class TFReservoir:
    """GPU reservoir: Chebyshev propagation of exp(-iH*dt) for a batch of images."""

    def __init__(self, reservoir: RydbergReservoir, cfg: QRCConfig):
        import tensorflow as tf

        self.tf = tf
        self.n = cfg.n_qubits
        self.dim = reservoir.dim
        self.n_steps = cfg.n_steps
        self.dt = cfg.dt
        self.k = cfg.detuning_scale
        self.R = self.n * cfg.omega

        Hc = reservoir._H_fixed.tocoo()
        idx = np.stack([Hc.row, Hc.col], 1).astype(np.int64)
        self.Hsp = tf.sparse.reorder(
            tf.SparseTensor(idx, Hc.data.astype(np.complex64), [self.dim, self.dim])
        )
        self.Hdiag = tf.constant(reservoir._H_fixed.diagonal().real.astype(np.float32)[:, None])
        self.Z = tf.constant(reservoir._Z_eigs.astype(np.complex64))
        self.obs_mat = tf.constant(np.stack(reservoir._obs).astype(np.float32))
        self.embed_dim = reservoir.embed_dim

    def embed_batch(self, z_batch: np.ndarray, tol: float = CHEB_TOL) -> np.ndarray:
        from scipy.special import jv

        tf = self.tf
        dt, dim, B = self.dt, self.dim, z_batch.shape[0]
        zc = tf.constant(z_batch.astype(np.complex64))
        D = tf.transpose((-0.5 * self.k) * tf.matmul(zc, self.Z))
        full = self.Hdiag + tf.math.real(D)
        Emax = float(tf.reduce_max(full)) + self.R
        Emin = float(tf.reduce_min(full)) - self.R
        c, Delta = (Emax + Emin) / 2.0, (Emax - Emin) / 2.0

        x = Delta * dt
        K0 = int(np.ceil(x)) + 30
        ks = np.arange(K0 + 1)
        ak = (2.0 - (ks == 0)) * ((-1j) ** ks) * jv(ks, x)
        keep = np.where(np.abs(ak) > tol)[0]
        ak = (ak[:int(keep[-1]) + 1] * np.exp(-1j * c * dt)).astype(np.complex64)

        cI = tf.constant(c, tf.complex64)
        invD = tf.constant(1.0 / Delta, tf.complex64)
        two = tf.constant(2.0, tf.complex64)

        def Htil(v):
            return (tf.sparse.sparse_dense_matmul(self.Hsp, v) + D * v - cI * v) * invD

        state = tf.constant(np.eye(1, dim, 0, dtype=np.complex64).T * np.ones((1, B), np.complex64))
        feats = []
        for _ in range(self.n_steps):
            T0, T1 = state, Htil(state)
            out = ak[0] * T0 + ak[1] * T1
            for kk in range(2, len(ak)):
                T2 = two * Htil(T1) - T0
                out = out + ak[kk] * T2
                T0, T1 = T1, T2
            state = out
            pr = tf.math.real(state * tf.math.conj(state))
            pr = pr / tf.reduce_sum(pr, axis=0, keepdims=True)
            feats.append(tf.matmul(self.obs_mat, pr))
        return tf.transpose(tf.concat(feats, 0)).numpy()


def compute_embeddings_gpu(tf_reservoir, z_scaled, cache_path, batch_size=BATCH_SIZE_QRC):
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)
    n = len(z_scaled)
    out = np.empty((n, tf_reservoir.embed_dim), dtype=np.float64)
    print(f"Computing {n} embeddings on GPU (batch={batch_size}) ...")
    t0 = time.perf_counter()
    for i in range(0, n, batch_size):
        out[i:i + batch_size] = tf_reservoir.embed_batch(z_scaled[i:i + batch_size])
        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, n)
            rate = (time.perf_counter() - t0) / done
            print(f"  {done}/{n}  ({rate:.3f}s/img, eta {rate*(n-done)/60:.1f} min)", flush=True)
    np.save(cache_path, out)
    print(f"Done in {(time.perf_counter()-t0)/60:.1f} min -> saved to {cache_path}")
    return out


# ----------------------------------------------------------------------
# 5/6. MLP training (subprocess, same as the notebook — GPU memory released between runs)
# ----------------------------------------------------------------------

def train_mlp(out_path: Path, x, y, xval, yval, tag: str):
    if out_path.exists():
        print(f"{tag}: cached model at {out_path} — skipping training")
        return
    ad = MODEL_DIR / "_arrays"
    ad.mkdir(parents=True, exist_ok=True)
    np.save(ad / f"{tag}_x.npy", x)
    np.save(ad / f"{tag}_y.npy", y)
    np.save(ad / f"{tag}_xv.npy", xval)
    np.save(ad / f"{tag}_yv.npy", yval)
    cmd = [sys.executable, "train_scripts/train_readout.py",
           "--x", str(ad / f"{tag}_x.npy"), "--y", str(ad / f"{tag}_y.npy"),
           "--xval", str(ad / f"{tag}_xv.npy"), "--yval", str(ad / f"{tag}_yv.npy"),
           "--out", str(out_path),
           "--epochs", str(EPOCHS), "--batch", str(BATCH_SIZE),
           "--lr", str(LEARNING_RATE), "--dropout", str(DROPOUT_RATE),
           "--patience", str(PATIENCE)]
    print(f"{tag}: training in subprocess:\n  {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"{tag}: saved to {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sigma", type=float, default=0.7, help="speckle noise std")
    ap.add_argument("--d-qrc", type=int, default=18, help="qubits / PCA components fed to reservoir")
    ap.add_argument("--batch-qrc", type=int, default=BATCH_SIZE_QRC, help="GPU embedding batch size")
    args = ap.parse_args()

    sigma, d_qrc = args.sigma, args.d_qrc

    random.seed(SEED)
    np.random.seed(SEED)

    import tensorflow as tf
    tf.random.set_seed(SEED)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset
    download_eurosat()
    X, y = load_dataset()
    X_train, X_val, X_test = split_dataset(X, y)

    # 2. Noise (same per-split seeds as the notebook)
    X_train_noisy = apply_noise(X_train, sigma=sigma, seed=SEED)
    X_val_noisy = apply_noise(X_val, sigma=sigma, seed=SEED + 1)
    X_test_noisy = apply_noise(X_test, sigma=sigma, seed=SEED + 2)

    # 3. PCA fit on clean train, project noisy splits
    from sklearn.decomposition import PCA

    X_train_flat = X_train.reshape(len(X_train), -1)
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(X_train_flat)
    print(f"PCA({N_COMPONENTS}) cumulative variance: "
          f"{np.cumsum(pca.explained_variance_ratio_)[-1]:.4f}")

    z_train = pca.transform(X_train_noisy.reshape(len(X_train_noisy), -1))[:, :d_qrc]
    z_val = pca.transform(X_val_noisy.reshape(len(X_val_noisy), -1))[:, :d_qrc]
    z_test = pca.transform(X_test_noisy.reshape(len(X_test_noisy), -1))[:, :d_qrc]

    # Min-max scale to [0, 1] on train stats (reservoir detunings expect [0, 1])
    z_min, z_max = z_train.min(axis=0), z_train.max(axis=0)

    def scale_z(z, eps=1e-8):
        return np.clip((z - z_min) / (z_max - z_min + eps), 0.0, 1.0).astype(np.float64)

    z_train_scaled, z_val_scaled, z_test_scaled = map(scale_z, (z_train, z_val, z_test))

    # 4. QRC embeddings (GPU, cached)
    embed_dir = MODEL_DIR / f"sigma{sigma}_embeddings_d{d_qrc}"
    embed_dir.mkdir(parents=True, exist_ok=True)
    cache = {s: embed_dir / f"{s}.npy" for s in ("train", "val", "test")}

    if all(p.exists() for p in cache.values()):
        print("All embeddings cached — skipping reservoir build.")
        R_train, R_val, R_test = (np.load(cache[s]) for s in ("train", "val", "test"))
    else:
        cfg = QRCConfig(n_qubits=d_qrc)
        reservoir = RydbergReservoir(cfg)
        tf_reservoir = TFReservoir(reservoir, cfg)

        # GPU vs CPU smoke check
        n_smoke = 5
        cpu_ref = np.array([reservoir.transform_one(z) for z in z_train_scaled[:n_smoke]])
        gpu_check = tf_reservoir.embed_batch(z_train_scaled[:n_smoke])
        max_err = np.abs(gpu_check - cpu_ref).max()
        print(f"GPU vs CPU smoke — max abs error: {max_err:.2e}")
        assert max_err < 5e-3, "GPU reservoir disagrees with CPU reference beyond fp32 tolerance"

        R_train = compute_embeddings_gpu(tf_reservoir, z_train_scaled, cache["train"], args.batch_qrc)
        R_val = compute_embeddings_gpu(tf_reservoir, z_val_scaled, cache["val"], args.batch_qrc)
        R_test = compute_embeddings_gpu(tf_reservoir, z_test_scaled, cache["test"], args.batch_qrc)

    print(f"Embeddings — train: {R_train.shape}, val: {R_val.shape}, test: {R_test.shape}")

    # 5. Standardise embeddings + train hybrid readout
    from sklearn.preprocessing import StandardScaler
    import joblib

    embed_scaler = StandardScaler()
    R_train_std = embed_scaler.fit_transform(R_train).astype(np.float32)
    R_val_std = embed_scaler.transform(R_val).astype(np.float32)
    joblib.dump(embed_scaler, MODEL_DIR / f"sigma{sigma}_embed_scaler_d{d_qrc}.joblib")

    y_train = X_train.reshape(len(X_train), -1).astype(np.float32)
    y_val = X_val.reshape(len(X_val), -1).astype(np.float32)

    train_mlp(MODEL_DIR / f"sigma{sigma}_readout_d{d_qrc}.keras",
              R_train_std, y_train, R_val_std, y_val, tag="hybrid")

    # 6. Classical baseline: standardised PCA components -> same MLP
    pca_scaler = StandardScaler()
    Zb_train = pca_scaler.fit_transform(z_train_scaled).astype(np.float32)
    Zb_val = pca_scaler.transform(z_val_scaled).astype(np.float32)

    train_mlp(MODEL_DIR / f"sigma{sigma}_baseline_pca{d_qrc}.keras",
              Zb_train, y_train, Zb_val, y_val, tag="baseline")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

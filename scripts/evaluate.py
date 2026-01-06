import os
from os.path import join
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append("../")

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

import argparse, math, warnings, json, os, random
from pathlib import Path
from typing   import List, Dict, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import skimage.io
import torchxrayvision as xrv
from vendi_score import vendi
from torchmetrics.image.fid import FrechetInceptionDistance


TARGET_COLS = ["Cardiomegaly", "Lung Opacity", "Edema", "No Finding", "Pneumothorax", "Pleural Effusion"]

from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.stats import entropy
from PIL import Image

class ISDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

@torch.no_grad()
def compute_inception_score(image_paths, batch_size=32, splits=10, device="cuda"):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    dataset = ISDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    preds = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        with torch.cuda.amp.autocast():
            logits = model(batch)
        preds.append(F.softmax(logits, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k+1) * (len(preds) // splits)]
        py = np.mean(part, axis=0)
        scores = [entropy(p, py) for p in part]
        split_scores.append(np.exp(np.mean(scores)))

    return float(np.mean(split_scores)), float(np.std(split_scores))

# -------------------------
# 1. Dataset + Preprocessing
# -------------------------
class XRVPathDataset(torch.utils.data.Dataset):
    """Load image → normalize → crop → resize(224) → tensor[-1024,1024]."""
    def __init__(self, paths: List[Path]):
        self.paths = list(map(str, paths))
        self.crop_resize = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])

    def __len__(self): 
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx].replace("CheXpert-v1.0/", "CheXpert-v1.0_1024x1024/")  # fix path
        img = skimage.io.imread(path)
        img = xrv.datasets.normalize(img, 255)
        print(img.max(), img.min(), img.shape, img.dtype)
        if img.ndim == 2:              
            img = img[None]
        elif img.shape[-1] == 3:      
            img = img.mean(2, keepdims=True).transpose(2,0,1)
        elif img.shape[0] == 1:        
            pass
        img = self.crop_resize(img)
        return torch.from_numpy(img)

def make_loader(paths, bs, nw):
    return DataLoader(XRVPathDataset(paths), bs, False,
                      num_workers=nw, pin_memory=True)
    
def dense121_feature_extractor(device="cpu") -> nn.Module:
    base = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
    class Feat(nn.Module):
        num_features = 1024
        def forward(self, x):
            if x.size(1) == 3: x = x.mean(1, keepdim=True)
            z = F.relu(base.features(x), inplace=True)
            return F.adaptive_avg_pool2d(z, 1).flatten(1)
    return Feat()


@torch.no_grad()
def extract_latents(paths, batch, device="cuda") -> np.ndarray:
    feat = dense121_feature_extractor(device)
    loader = make_loader(paths, batch, 4)
    out = []
    for xb in loader:
        out.append(feat(xb.to(device)).cpu())
    return torch.cat(out,0).numpy().astype("float32") 

@torch.inference_mode()
def fid_score(real_paths, fake_paths, batch, device="cuda") -> float:
    feat = dense121_feature_extractor(device)
    metric = FrechetInceptionDistance(
        feature=feat, normalize=True, reset_real_features=True,
        input_img_size=(1,224,224)
    ).to(device)
    dl_r = make_loader(real_paths, batch, 4)
    dl_f = make_loader(fake_paths, batch, 4)
    for xb in tqdm(dl_r): metric.update(xb.to(device), real=True)
    for xb in tqdm(dl_f): metric.update(xb.to(device), real=False)
    return float(metric.compute())

def estimate_kl_mc_kde(Zp, Zq, bandwidth=1.0):
    kde_p = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(Zp)
    kde_q = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(Zq)

    log_p = kde_p.score_samples(Zp)  # log p(z)
    log_q = kde_q.score_samples(Zp)  # log q(z) for z ~ P

    kl = np.mean(log_p - log_q)
    return kl

def create_subset_df(df, combdist_df, subset_size):
    # Merge the main data with the combdist data
    df = df.merge(combdist_df[['combo_key', 'proportion']], on='combo_key', how='inner')

    # For each combination, sample proportionally
    df_subset = (
        df
        .groupby('combo_key', group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), int(subset_size * x['proportion'].iloc[0])), random_state=42))
    )

    # Drop helper column
    df_subset = df_subset.drop(columns=['combo_key'])
    
    return df_subset
    
def compute_per_class_metrics(df_real, df_fake, args, results):
    for disease in TARGET_COLS:
        real_subset = df_real[df_real[disease] == 1]["Path"].dropna().tolist()
        fake_subset = df_fake[df_fake[disease] == 1]["Path"].dropna().tolist()

        real_paths = [join(args.chexpert_root, p) for p in real_subset]
        fake_paths = [join(args.gen_root, p) for p in fake_subset]

        if len(real_paths) < 50 or len(fake_paths) < 50:
            print(f"Skipping {disease} (not enough samples)")
            continue

        print(f"\n--- {disease} ---")
        # try:
        #     fid_val = fid_score(real_sample, fake_sample, args.batch, args.device)
        #     print(f"FID ({disease}): {fid_val:.4f}")
        #     results.append(dict(prompt=disease, subset="class", metric="FID", value=fid_val))
        # except Exception as e:
        #     print(f"[{disease}] FID failed: {e}")

        # try:
        #     # is_mean_imnet, is_std_imnet = compute_inception_score(fake_sample, batch_size=args.batch, device=args.device)
        #     # is_mean_chex, is_std_chex = compute_chexpert_inception_score(fake_paths, batch_size=args.batch, device=args.device)


        #     # print(f"Inception Score (ImageNet, {disease}): {is_mean_imnet:.4f} ± {is_std_imnet:.4f}")
        #     # print(f"Inception Score (CheXpert, {disease}): {is_mean_chex:.4f} ± {is_std_chex:.4f}")
        #     # results.append(dict(prompt=disease, subset="class", metric="IS_ImageNet", value=is_mean_imnet))
        #     # results.append(dict(prompt=disease, subset="class", metric="IS_CheXpert", value=is_mean_chex))

        # except Exception as e:
        #     print(f"[{disease}] IS failed: {e}")

@torch.no_grad()
def compute_chexpert_inception_score(image_paths, batch_size=32, splits=10, device="cuda"):
    model = xrv.models.DenseNet(weights="densenet121-res224-chex").to(device)
    model.eval()

    dataset = XRVPathDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    preds = []
    for xb in tqdm(loader, desc="CheXpert-IS"):
        xb = xb.to(device)
        if xb.shape[1] == 3:
            xb = xb.mean(1, keepdim=True)
        pred = model(xb).sigmoid().cpu().numpy()
        preds.append(pred)

    preds = np.concatenate(preds, axis=0)

    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits)]
        py = np.mean(part, axis=0)
        scores = [entropy(p, py + 1e-6) for p in part]
        split_scores.append(np.exp(np.mean(scores)))

    return float(np.mean(split_scores)), float(np.std(split_scores))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chexpert_csv",  required=True)
    ap.add_argument("--chexpert_root", required=True)
    ap.add_argument("--gen_csv",     required=True)
    ap.add_argument("--gen_root",      required=True)
    ap.add_argument("--out_dir",       default="eval_out")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_components", type=int, default=100)
    ap.add_argument("--bandwidth", type=float, default=1.0)
    args = ap.parse_args()

    model_name = Path(args.gen_root).stem          # e.g.  logs‑21001 → logs‑21001
    out_dir = Path(args.out_dir)
    lat_dir = out_dir / f"latents-{model_name}" / "dense224"
    lat_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
   
    subset_sizes = [640, 320, 160, 80, 40]
    
    subset_pool: dict[int, list[np.ndarray]] = {n: [] for n in subset_sizes}
    
    # Load the combdist CSV
    # df_combdist = pd.read_csv("datasets/chexpert/chexpert_combdist_train.csv")
    # Normalize counts to get proportions
    # df_combdist['proportion'] = df_combdist['count'] / df_combdist['count'].sum()
    
    # Group the main data by label combination
    # df_combdist['combo_key'] = df_combdist[TARGET_COLS].astype(str).agg('_'.join, axis=1)


    df_real = pd.read_csv(args.chexpert_csv)
    # df_real['combo_key'] = df_real[TARGET_COLS].astype(str).agg('_'.join, axis=1)
    # # Define total size of subset (e.g., 10,000 samples)
    # desired_total = 2000
    # df_real = create_subset_df(df_real, df_combdist, desired_total)
    real_paths = df_real["Path"].values
    real_paths = [join(args.chexpert_root, p) for p in real_paths]

    df_fake = pd.read_csv(args.gen_csv)
    # df_fake['combo_key'] = df_fake[TARGET_COLS].astype(str).agg('_'.join, axis=1)
    # desired_total = 2000
    # df_fake = create_subset_df(df_fake, df_combdist, desired_total)
    fake_paths = df_fake["Path"].values
    fake_paths = [join(args.gen_root, p) for p in fake_paths]

    # ======================  GLOBAL ("ALL") METRICS  =============================

    # --- Inception Score ---
    # print("Computing Inception Score...")
    # is_mean_cxpt, is_std_cxpt = compute_chexpert_inception_score(fake_paths, batch_size=args.batch, device=args.device)

    # is_mean_imnet, is_std_imnet = compute_inception_score(fake_paths, batch_size=args.batch, device=args.device)
    
    # results.append(dict(prompt="ALL", subset="ALL", metric="IS_ImageNet", value=is_mean_imnet))
    # results.append(dict(prompt="ALL", subset="ALL", metric="IS_CheXpert", value=is_mean_cxpt))
    # print(f"Inception Score (ImageNet): {is_mean_imnet:.4f} ± {is_std_imnet:.4f}")
    # print(f"Inception Score (CheXpert): {is_mean_cxpt:.4f} ± {is_std_cxpt:.4f}")
    
    # 1)  FID across the whole validation split
    print("lenghts: ", len(real_paths), len(fake_paths))
    # fid_all = fid_score(real_paths, fake_paths, args.batch, args.device)
    # print(f"\nFID (all prompts): {fid_all}")
    # results.append(dict(prompt="ALL", subset="ALL", metric="FID", value=fid_all))

    # 2)  Vendi for each subset size aggregated across prompts
    for n in subset_sizes:
        print(f"\nComputing Vendi for first {n} prompts...")
        if not subset_pool[n]:
            continue
        z_concat = np.concatenate(subset_pool[n], axis=0)   # (P*n, 1024)
        np.save(lat_dir / f"ALL_{n}.npy", z_concat)         # cache for later use
        vendi_all = float(vendi.score_dual(z_concat))
        print(f"Vendi(all, first {n}/prompt): {vendi_all}")
        results.append(dict(prompt="ALL", subset=str(n),
                            metric="Vendi", value=vendi_all))

    # print(f"\nFID (all prompts): {fid_all:6.2f}")

    
    # 3) Per-class FID and Inception Score
    results = compute_per_class_metrics(df_real, df_fake, args, results)

    for n in subset_sizes:
        if subset_pool[n]:
            print(f"Vendi(all, first {n}/prompt): {res_df[(res_df.prompt=='ALL')&(res_df.subset==str(n))].value.iloc[0]:.4f}")

    
    res_df = pd.DataFrame(results)
    csv_path = out_dir / f"results-{model_name}.csv"
    res_df.to_csv(csv_path, index=False)

    print(f"Latents saved to {lat_dir}")
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()

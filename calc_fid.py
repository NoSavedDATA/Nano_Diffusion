import numpy as np
from scipy import linalg



def load_stats(npz_path):
    data = np.load(npz_path)
    return data['mu'], data['sigma']


def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        print("Warning: fid sqrtm not finite, adding eps to diag")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid



if __name__=="__main__":

    mu1, sigma1 = load_stats("fid_dists/dit.npz")
    mu2, sigma2 = load_stats("fid_dists/imagenet.npz")

    fid_score = compute_fid(mu1, sigma1, mu2, sigma2)
    print("ImageNet DIT FID:", fid_score)


    mu1, sigma1 = load_stats("fid_dists/games.npz")
    mu2, sigma2 = load_stats("fid_dists/sd15_finetune.npz")

    fid_score = compute_fid(mu1, sigma1, mu2, sigma2)
    print("Games dataset finetuned Stable Diffusion FID:", fid_score)

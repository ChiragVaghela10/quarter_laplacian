import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


def save_filter_results(base_img: np.ndarray, qlf_img: np.ndarray, laplace_img: np.ndarray, result_path: Path) -> None:
    imgs = [base_img, qlf_img, laplace_img]
    titles = ['Orignal Image', 'QLF Filter', 'Laplace Filter']
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(ncols=3, wspace=0)
    axes = gs.subplots(sharex=True, sharey=True)

    for i, (img, title) in enumerate(zip(imgs, titles)):
        axes[i].set_title(title)
        axes[i].imshow(img, cmap='gray')

    fig.suptitle('Comparison of Quarter Laplacian and Laplacian Filter', y=0.98)
    fig.tight_layout()
    fig.savefig(result_path)

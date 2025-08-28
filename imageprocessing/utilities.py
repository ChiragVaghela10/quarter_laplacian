import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from matplotlib.gridspec import GridSpec


def save_filter_comparison(base_img: np.ndarray,
                           qlf_results: dict,
                           laplace_results: dict,
                           result_path: Path) -> None:
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])

    # Base image spanning both rows
    ax_base = fig.add_subplot(gs[:, 0])
    ax_base.imshow(base_img, cmap="gray")
    ax_base.set_title("Base Image")
    ax_base.axis("off")

    # QLF row
    ax_qlf = [fig.add_subplot(gs[0, i]) for i in range(1, 4)]
    for ax, img, it in zip(ax_qlf, list(qlf_results.values()), list(qlf_results.keys())):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"QLF (iters={it})")
        ax.axis("off")

    # Laplacian row
    ax_lap = [fig.add_subplot(gs[1, i]) for i in range(1, 4)]
    for ax, img, it in zip(ax_lap, list(laplace_results.values()), list(laplace_results.keys())):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Laplacian (iters={it})")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(result_path)
    plt.close(fig)


def plot_metric_comparison(metric_dict, result_path: Path):
    metrics = list(metric_dict.keys())
    qlf_values = [metric_dict[m][0] for m in metrics]
    lap_values = [metric_dict[m][1] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width/2 for i in x], qlf_values, width, label='QLF')
    ax.bar([i + width/2 for i in x], lap_values, width, label='Laplacian')

    ax.set_ylabel('Score')
    ax.set_title('QLF vs Laplacian Comparison')
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(result_path)

def plot_filters_against_alphas(filters_performance: dict, alphas: np.ndarray, img_path: Path) -> None:
    plt.suptitle('Low Light Enhancement', y=0.9)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=False)
    ax1, ax2, ax3 = axes
    ax1.plot(alphas, filters_performance['qlf_psnrs'], marker='o', label='QLF')
    ax1.plot(alphas, filters_performance['laplace_psnrs'], marker='s', label='Laplacian')
    ax1.set_title('PSNR (Y)')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('dB')
    ax1.grid(True)
    ax1.legend()
    ax2.plot(alphas, filters_performance['qlf_ssims'], marker='o', label='QLF')
    ax2.plot(alphas, filters_performance['laplace_ssims'], marker='s', label='Laplacian')
    ax2.set_title('SSIM (Y)')
    ax2.set_xlabel('alpha')
    ax2.grid(True)
    ax2.legend()
    ax3.plot(alphas, filters_performance['qlf_epis'], marker='o', label='QLF')
    ax3.plot(alphas, filters_performance['laplace_epis'], marker='s', label='Laplacian')
    ax3.set_title('EPI (Canny Dice on Y)')
    ax3.set_xlabel('alpha')
    ax3.grid(True)
    ax3.legend()
    plt.tight_layout()
    plt.savefig(str(img_path))
    plt.close()


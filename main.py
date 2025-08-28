import os
import cv2 as cv
# import opendatasets as od
from statistics import mean
from pathlib import Path

import numpy as np
from tqdm import tqdm

from constants import ITERATIONS, TIME_STEP
from imageprocessing.filters import QuarterLaplacian, LaplacianFilter
from imageprocessing.experiments import LowLightEnhancement
from imageprocessing.quantitative_analysis import QuantitativeAnalysis
from imageprocessing.utilities import plot_metric_comparison, save_filter_comparison, plot_filters_against_alphas

# from constants import LOW_LIGHT_IMG_DATASET
# od.download(LOW_LIGHT_IMG_DATASET)

ROOT_PATH = Path(__file__).parent
IMG_DIR = ROOT_PATH / 'img'
LOW_LIGHT_IMG_DIR1 = ROOT_PATH / 'lol-dataset/lol_dataset/our485/low'
HIGH_LIGHT_IMG_DIR1 = ROOT_PATH / 'lol-dataset/lol_dataset/our485/high'
# LOW_LIGHT_IMG_DIR2 = ROOT_PATH / 'lol-dataset/lol_dataset/eval15/low'
# LOW_LIGHT_IMG_DIR2 = ROOT_PATH / 'lol-dataset/lol_dataset/eval15/high'

cameraman_img = cv.imread(str(IMG_DIR / 'cameraman.png'), cv.IMREAD_UNCHANGED)
assert cameraman_img is not None, "file could not be read"
print(f'Loaded image type: {cameraman_img.dtype} and shape: {cameraman_img.shape}')

low_light_img = cv.imread(str(IMG_DIR / 'low_light.png'), cv.IMREAD_UNCHANGED)
assert low_light_img is not None, "low-light image could not be read"
normal_img = cv.imread(str(IMG_DIR / 'normal.png'), cv.IMREAD_UNCHANGED)
assert normal_img is not None, "base image not be read"
print(f'Loaded low-light image type: {low_light_img.dtype} and shape: {low_light_img.shape}')

# Visual comparison of QLF and Laplace Filter for different iterations
qlf = QuarterLaplacian()
qlf_imgs = {f'{i}': qlf.apply_filter(U=cameraman_img, iterations=i) for i in [10, 100, 1000]}

laplacian = LaplacianFilter()
laplacian_imgs = {f'{i}': laplacian.apply_filter(U=cameraman_img, iterations=i) for i in [10, 100, 1000]}

save_filter_comparison(base_img=cameraman_img, qlf_results=qlf_imgs,
                       laplace_results=laplacian_imgs, result_path=IMG_DIR / 'comparison.png')

# Filter performance comparison on enhancement of low-light images for different step size (time-step)
low_light_experiment = LowLightEnhancement()
alphas = np.arange(0.1, 1.1, 0.1)
low_light_qlf_result = [low_light_experiment.enhance(
    low=low_light_img,
    ref=normal_img,
    filter=qlf,
    iterations=ITERATIONS,
    alpha=alpha
)['enhanced'] for alpha in alphas]
low_light_laplace_result = [low_light_experiment.enhance(
    low=low_light_img,
    ref=normal_img,
    filter=laplacian,
    iterations=ITERATIONS,
    alpha=alpha
)['enhanced'] for alpha in alphas]

quant_analysis = QuantitativeAnalysis()
qlf_psnrs, qlf_ssims, qlf_epis = [], [], []
laplace_psnrs, laplace_ssims, laplace_epis = [], [], []
for qlf_result, laplace_result in zip(low_light_qlf_result, low_light_laplace_result):
    qa_result = quant_analysis.analyse(base_img=normal_img, qlf_img=qlf_result, laplace_img=laplace_result)
    qlf_psnrs.append(qa_result['qlf_psnr'])
    qlf_ssims.append(qa_result['qlf_ssim'])
    qlf_epis.append(qa_result['qlf_epi'])
    laplace_psnrs.append(qa_result['laplace_psnr'])
    laplace_ssims.append(qa_result['laplace_ssim'])
    laplace_epis.append(qa_result['laplace_epi'])

plot_filters_against_alphas(
    filters_performance={
        "qlf_psnrs": qlf_psnrs,
        "laplace_psnrs": laplace_psnrs,
        "qlf_ssims": qlf_ssims,
        "laplace_ssims": laplace_ssims,
        "qlf_epis": qlf_epis,
        "laplace_epis": laplace_epis,
    },
    alphas=alphas,
    img_path=IMG_DIR / 'filters_against_alphas.png'
)

# Filter performance comparison on LOL dataset
low_light_files = sorted([os.path.join(LOW_LIGHT_IMG_DIR1, fname) for fname in os.listdir(LOW_LIGHT_IMG_DIR1)])
normal_light_files = sorted([os.path.join(LOW_LIGHT_IMG_DIR1, fname) for fname in os.listdir(HIGH_LIGHT_IMG_DIR1)])
quant_metrics = {
    "qlf_psnr": [],
    "qlf_ssim": [],
    "qlf_epi": [],
    "laplace_psnr": [],
    "laplace_ssim": [],
    "laplace_epi": [],
}
for low, ref in tqdm(zip(low_light_files, normal_light_files), total=len(low_light_files), desc='Enhancing Images'):
    low = cv.imread(low, cv.IMREAD_UNCHANGED)
    ref = cv.imread(ref, cv.IMREAD_UNCHANGED)
    qlf_result = low_light_experiment.enhance(
        low=low, ref=ref, filter=qlf, iterations=ITERATIONS, alpha=TIME_STEP
    )['enhanced']
    laplacian_result = low_light_experiment.enhance(
        low=low, ref=ref, filter=laplacian, iterations=ITERATIONS, alpha=TIME_STEP
    )['enhanced']
    qa_result = quant_analysis.analyse(base_img=ref, qlf_img=qlf_result, laplace_img=laplacian_result)
    quant_metrics['qlf_psnr'].append(qa_result['qlf_psnr'])
    quant_metrics['qlf_ssim'].append(qa_result['qlf_ssim'])
    quant_metrics['qlf_epi'].append(qa_result['qlf_epi'])
    quant_metrics['laplace_psnr'].append(qa_result['laplace_psnr'])
    quant_metrics['laplace_ssim'].append(qa_result['laplace_ssim'])
    quant_metrics['laplace_epi'].append(qa_result['laplace_epi'])

avg_qlf_psnr = mean(quant_metrics['qlf_psnr'])
avg_laplace_psnr = mean(quant_metrics['laplace_psnr'])
avg_qlf_ssim = mean(quant_metrics['qlf_ssim'])
avg_laplace_ssim = mean(quant_metrics['laplace_ssim'])
avg_qlf_epi = mean(quant_metrics['qlf_epi'])
avg_laplace_epi = mean(quant_metrics['laplace_epi'])

print("\n=== Average Quantitative Results (via quant_result) ===")
print(f"Avg QLF PSNR:     {avg_qlf_psnr:.2f}")
print(f"Avg Laplace PSNR: {avg_laplace_psnr:.2f}")
print(f"Avg QLF SSIM:     {avg_qlf_ssim:.2f}")
print(f"Avg Laplace SSIM: {avg_laplace_ssim:.2f}")
print(f"Avg QLF EPI:      {avg_qlf_epi:.2f}")
print(f"Avg Laplace EPI:  {avg_laplace_epi:.2f}")

metric_results = {
    "PSNR": (avg_qlf_psnr, avg_laplace_psnr),
    "SSIM": (avg_qlf_ssim, avg_laplace_ssim),
    "EPI":  (avg_qlf_epi, avg_laplace_epi),
}

plot_metric_comparison(metric_results, result_path=IMG_DIR / 'metric_comparison.png')

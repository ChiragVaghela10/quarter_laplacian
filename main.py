import cv2 as cv
# import opendatasets as od
from statistics import mean
from pathlib import Path

import numpy as np
from flatbuffers.packer import float32
from tqdm import tqdm

from constants import DARKEN_FACTOR
from imageprocessing.filters import QuarterLaplacian, LaplacianFilter
from imageprocessing.experiments import LowLightEnhancement
from imageprocessing.low_light_simulation import DegradeSimulation
from imageprocessing.quantitative_analysis import QuantitativeAnalysis
from imageprocessing.utilities import plot_metric_comparison, save_filter_comparison, plot_filters_against_alphas

# from constants import LOW_LIGHT_IMG_DATASET
# od.download(LOW_LIGHT_IMG_DATASET)

ROOT_PATH = Path(__file__).parent
IMG_DIR = ROOT_PATH / 'img'
# INDEED_DATASET = ROOT_PATH / 'indeed_dataset_downsized/'
# SHENZHEN_DATASET = ROOT_PATH / 'shenzhen/'
# LOW_LIGHT_IMG_DIR1 = ROOT_PATH / 'lol-dataset/lol_dataset/our485/low'
# HIGH_LIGHT_IMG_DIR1 = ROOT_PATH / 'lol-dataset/lol_dataset/our485/high'
# LOW_LIGHT_IMG_DIR2 = ROOT_PATH / 'lol-dataset/lol_dataset/eval15/low'

cameraman_img = cv.imread(str(IMG_DIR / 'cameraman.png'), cv.IMREAD_UNCHANGED)
assert cameraman_img is not None, "file could not be read"
print(f'Loaded image type: {cameraman_img.dtype} and shape: {cameraman_img.shape}')

low_light_img = cv.imread(str(IMG_DIR / 'low_light.png'), cv.IMREAD_UNCHANGED)
assert low_light_img is not None, "low-light image could not be read"
normal_img = cv.imread(str(IMG_DIR / 'normal.png'), cv.IMREAD_UNCHANGED)
assert normal_img is not None, "base image not be read"
print(f'Loaded low-light image type: {low_light_img.dtype} and shape: {low_light_img.shape}')

# Comparison of QLF and Laplace Filter
qlf = QuarterLaplacian()
qlf_imgs = {f'{i}': qlf.apply_filter(U=cameraman_img, iterations=i) for i in [10, 100, 1000]}

laplacian = LaplacianFilter()
laplacian_imgs = {f'{i}': laplacian.apply_filter(U=cameraman_img, iterations=i) for i in [10, 100, 1000]}

save_filter_comparison(base_img=cameraman_img, qlf_results=qlf_imgs,
                       laplace_results=laplacian_imgs, result_path=IMG_DIR / 'comparison.png')

# Low Light Image Enhancement Experiment
low_light_experiment = LowLightEnhancement()
alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
low_light_qlf_result = [low_light_experiment.enhance(low=low_light_img, ref=normal_img, filter=qlf, alpha=alpha)['enhanced'] for alpha in alphas]
low_light_laplace_result = [low_light_experiment.enhance(low=low_light_img, ref=normal_img, filter=laplacian, alpha=alpha)['enhanced'] for alpha in alphas]

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

# Quantitative analysis by comparing PSNR, SSIM, EPI metric values between QLF and baseline Laplacian filter
# degradation = DegradeSimulation()
# low_light_cameraman_img = degradation.simulate_low_light(base_image=cameraman_img, darken_factor=DARKEN_FACTOR,
#                                                          blur_sigma=0, noise_sigma_pre=0)    # Try with diff sigmas
# restored_cameraman_img = degradation.restore_exposure(low_light_image01=low_light_cameraman_img,
#                                                       darken_factor=DARKEN_FACTOR)
# qlf_img = qlf.apply_filter(U=restored_cameraman_img, iterations=10)
# laplace_img = laplacian.apply_filter(U=restored_cameraman_img, iterations=10)
# cameraman_quant_analysis = QuantitativeAnalysis(base_img=cameraman_img, qlf_img=qlf_img, laplace_img=laplace_img)
# cameraman_analysis_result = cameraman_quant_analysis.analyse()
# cameraman_quant_analysis.show_results(result=cameraman_analysis_result)
# plot_metric_comparison(
#     metric_dict={
#         "PSNR": (cameraman_analysis_result['qlf_psnr'], cameraman_analysis_result['laplace_psnr']),
#         "SSIM": (cameraman_analysis_result['qlf_ssim'], cameraman_analysis_result['laplace_ssim']),
#         "EPI":  (cameraman_analysis_result['qlf_epi'], cameraman_analysis_result['laplace_epi']),
#     },
#     result_path=IMG_DIR / 'metric_comparison_on_cameraman_img.png'
# )

# import os
# low_light_files = sorted(os.listdir(LOW_LIGHT_IMG_DIR1))
# quant_metrics = []
# for file_name in tqdm(low_light_files, # sorted(INDEED_DATASET.glob('*.*')),
#                        desc='Enhancing Images'):
#     low_light_img = cv.imread(str(LOW_LIGHT_IMG_DIR1 / file_name), cv.IMREAD_GRAYSCALE)
#     high_light_img = cv.imread(str(HIGH_LIGHT_IMG_DIR1 / file_name), cv.IMREAD_GRAYSCALE)
#     low_light_exp_result = low_light_experiment.enhance(image=low_light_img)
#     quant_analysis = QuantitativeAnalysis(
#         base_img=high_light_img,
#         qlf_img=low_light_exp_result['qlf'],
#         laplace_img=low_light_exp_result['laplace'],
#     )
#     quant_analysis_result = quant_analysis.analyse()
#     quant_metrics.append(quant_analysis_result)
#
# avg_qlf_psnr = mean(res['qlf_psnr'] for res in quant_metrics)
# avg_laplace_psnr = mean(res['laplace_psnr'] for res in quant_metrics)
# avg_qlf_ssim = mean(res['qlf_ssim'] for res in quant_metrics)
# avg_laplace_ssim = mean(res['laplace_ssim'] for res in quant_metrics)
# avg_qlf_epi = mean(res['qlf_epi'] for res in quant_metrics)
# avg_laplace_epi = mean(res['laplace_epi'] for res in quant_metrics)
#
# print("\n=== Average Quantitative Results (via quant_result) ===")
# print(f"Avg QLF PSNR:     {avg_qlf_psnr:.2f}")
# print(f"Avg Laplace PSNR: {avg_laplace_psnr:.2f}")
# print(f"Avg QLF SSIM:     {avg_qlf_ssim:.2f}")
# print(f"Avg Laplace SSIM: {avg_laplace_ssim:.2f}")
# print(f"Avg QLF EPI:      {avg_qlf_epi:.2f}")
# print(f"Avg Laplace EPI:  {avg_laplace_epi:.2f}")
#
# metric_results = {
#     "PSNR": (avg_qlf_psnr, avg_laplace_psnr),
#     "SSIM": (avg_qlf_ssim, avg_laplace_ssim),
#     "EPI":  (avg_qlf_epi, avg_laplace_epi),
# }
#
# plot_metric_comparison(metric_results, result_path=IMG_DIR / 'metric_comparison.png')

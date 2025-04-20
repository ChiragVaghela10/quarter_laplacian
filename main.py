import cv2 as cv
import numpy as np
import opendatasets as od
from statistics import mean
from pathlib import Path
from tqdm import tqdm

from imageprocessing.filters import QuarterLaplacian
from imageprocessing.experiments import QuantitativeAnalysis, LowLightEnhancement
from imageprocessing.utilities import save_filter_results

from constants import LOW_LIGHT_IMG_DATASET

# od.download(LOW_LIGHT_IMG_DATASET)

ROOT_PATH = Path(__file__).parent
LOW_LIGHT_IMG_DIR1 = ROOT_PATH / 'lol-dataset/lol_dataset/our485/low'
LOW_LIGHT_IMG_DIR2 = ROOT_PATH / 'lol-dataset/lol_dataset/eval15/low'
CAMERAMAN_IMG_PATH = ROOT_PATH / 'img'

cameraman_img = cv.imread(str(CAMERAMAN_IMG_PATH / 'cameraman.png'), cv.IMREAD_GRAYSCALE)
assert cameraman_img is not None, "file could not be read"

low_light_img = cv.imread(str(CAMERAMAN_IMG_PATH / 'low_light.png'), cv.IMREAD_GRAYSCALE)
assert low_light_img is not None, "file could not be read"

# Comparison of QLF and Laplace Filter
qlf = QuarterLaplacian()
qlf_cameraman_img = qlf.apply_filter(U=cameraman_img.copy())

# Apply Laplacian Filter
laplace_cameraman_img = np.zeros_like(cameraman_img)
cv.Laplacian(src=cameraman_img, ddepth=-1, dst=laplace_cameraman_img, ksize=3,
             borderType=cv.BORDER_REPLICATE)
save_filter_results(base_img=cameraman_img, qlf_img=qlf_cameraman_img, laplace_img=laplace_cameraman_img,
                    result_path=CAMERAMAN_IMG_PATH / 'qlf_vs_std_laplace.png')

# Quantitative analysis by comparing PSNR, SSIM, EPI metrics between QLF and baseline Laplacian filter
cameraman_quant_analysis = QuantitativeAnalysis(base_img=cameraman_img, qlf_img=qlf_cameraman_img,
                                                laplace_img=laplace_cameraman_img)
cameraman_analysis_result = cameraman_quant_analysis.analyse()

print(f"QLF Cameraman PSNR: {cameraman_analysis_result['qlf_psnr']:.2f}, "
      f"Laplace Cameraman PSNR: {cameraman_analysis_result['laplace_psnr']:.2f}")

print(f"QLF Cameraman SSIM: {cameraman_analysis_result['qlf_ssim']:.2f}, "
      f"Laplace Cameraman SSIM: {cameraman_analysis_result['laplace_ssim']:.2f}")

print(f"QLF Cameraman EPI: {cameraman_analysis_result['qlf_epi']:.2f}, "
      f"Laplace Cameraman EPI: {cameraman_analysis_result['laplace_epi']:.2f}")

# Low Light Image Enhancement Experiment
low_light_experiment = LowLightEnhancement(qlf_filter=qlf)
quant_metrics = []
for image_path in tqdm(sorted(LOW_LIGHT_IMG_DIR1.glob('*.*')) + sorted(LOW_LIGHT_IMG_DIR1.glob('*.*')),
                       desc='Enhancing Images'):
    low_light_img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    low_light_exp_result = low_light_experiment.enhance(image=low_light_img)
    quant_analysis = QuantitativeAnalysis(
        base_img=low_light_img,
        qlf_img=low_light_exp_result['qlf'],
        laplace_img=low_light_exp_result['laplace'],
    )
    quant_analysis_result = quant_analysis.analyse()
    quant_metrics.append(quant_analysis_result)

avg_qlf_psnr = mean(res['qlf_psnr'] for res in quant_metrics)
avg_laplace_psnr = mean(res['laplace_psnr'] for res in quant_metrics)
avg_qlf_ssim = mean(res['qlf_ssim'] for res in quant_metrics)
avg_laplace_ssim = mean(res['laplace_ssim'] for res in quant_metrics)
avg_qlf_epi = mean(res['qlf_epi'] for res in quant_metrics)
avg_laplace_epi = mean(res['laplace_epi'] for res in quant_metrics)

print("\n=== Average Quantitative Results (via quant_result) ===")
print(f"Avg QLF PSNR:     {avg_qlf_psnr:.2f}")
print(f"Avg Laplace PSNR: {avg_laplace_psnr:.2f}")
print(f"Avg QLF SSIM:     {avg_qlf_ssim:.2f}")
print(f"Avg Laplace SSIM: {avg_laplace_ssim:.2f}")
print(f"Avg QLF EPI:      {avg_qlf_epi:.2f}")
print(f"Avg Laplace EPI:  {avg_laplace_epi:.2f}")

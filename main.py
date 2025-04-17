import cv2 as cv
import numpy as np
from pathlib import Path

from imageprocessing.filters import QuarterLaplacian
from imageprocessing.experiments import QuantitativeAnalysis, LowLightEnhancement

ROOT_PATH = Path(__file__).parent
img_path = ROOT_PATH / 'img'

cameraman_img = cv.imread(str(img_path / 'cameraman.png'), cv.IMREAD_GRAYSCALE)
assert cameraman_img is not None, "file could not be read"

low_light_img = cv.imread(str(img_path / 'low_light.png'), cv.IMREAD_GRAYSCALE)
assert low_light_img is not None, "file could not be read"

qlf = QuarterLaplacian()
qlf_cameraman_img = qlf.apply_filter(U=cameraman_img.copy())
laplace_cameraman_img = np.zeros_like(cameraman_img)
cv.Laplacian(src=cameraman_img, ddepth=-1, dst=laplace_cameraman_img, ksize=3,
             borderType=cv.BORDER_REPLICATE)

cv.imwrite(str(img_path / 'qlf_cameraman_img.png'), qlf_cameraman_img)
cv.imwrite(str(img_path / 'laplace_cameraman_img.png'), laplace_cameraman_img)

# Quantitative analysis by comparing PSNR, SSIM, EPI metrics between QLF and baseline Laplacian filter
cameraman_quant_experiment = QuantitativeAnalysis(base_img=cameraman_img, qlf_img=qlf_cameraman_img,
                                        laplace_img=laplace_cameraman_img)
cameraman_quant_experiment_result = cameraman_quant_experiment.analyse()

print(f"QLF Cameraman PSNR: {cameraman_quant_experiment_result['qlf_psnr']:.2f}, "
      f"Laplace Cameraman PSNR: {cameraman_quant_experiment_result['laplace_psnr']:.2f}")

print(f"QLF Cameraman SSIM: {cameraman_quant_experiment_result['qlf_ssim']:.2f}, "
      f"Laplace Cameraman SSIM: {cameraman_quant_experiment_result['laplace_ssim']:.2f}")

print(f"QLF Cameraman EPI: {cameraman_quant_experiment_result['qlf_epi']:.2f}, "
      f"Laplace Cameraman EPI: {cameraman_quant_experiment_result['laplace_epi']:.2f}")

# Low Light Image Enhancement Experiment
low_light_experiment = LowLightEnhancement(qlf_filter=qlf)

low_light_experiment_result = low_light_experiment.enhance(image=low_light_img)
low_light_experiment.save_experiment(results=low_light_experiment_result,
                                     img_path=img_path / 'low_light_exp_result.png')

low_light_quant_experiment = QuantitativeAnalysis(base_img=low_light_img,
                                                  qlf_img=low_light_experiment_result['qlf'],
                                                  laplace_img=low_light_experiment_result['laplace'])
low_light_quant_experiment_result = low_light_quant_experiment.analyse()

print(f"QLF Low Light PSNR: {low_light_quant_experiment_result['qlf_psnr']:.2f}, "
      f"Laplace Low Light PSNR: {low_light_quant_experiment_result['laplace_psnr']:.2f}")

print(f"QLF Low Light SSIM: {low_light_quant_experiment_result['qlf_ssim']:.2f}, "
      f"Laplace Low Light SSIM: {low_light_quant_experiment_result['laplace_ssim']:.2f}")

print(f"QLF Low Light EPI: {low_light_quant_experiment_result['qlf_epi']:.2f}, "
      f"Laplace Low Light EPI: {low_light_quant_experiment_result['laplace_epi']:.2f}")

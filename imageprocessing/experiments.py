import numpy as np
import cv2 as cv
from pathlib import Path
from matplotlib import pyplot as plt


from imageprocessing.filters import QuarterLaplacian, LaplacianFilter, Filter
from constants import MAX_PIXEL_VALUE


class LowLightEnhancement(object):
    """
    This class enhances low-light images by applying filters. Three different filters specified below are performed
    on the low light image for comparative study of the filters:
    1. Gamma Correction
    2. QLF
    3. Laplace Correction
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def estimate_gain_from_pair(low, ref, p_lo=5, p_hi=95, eps=1e-6, cap=(0.5, 8.0)):
        # 1) luminance (Y) in YCrCb
        Y_low = cv.cvtColor(low, cv.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
        Y_ref = cv.cvtColor(ref, cv.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0

        # 2) midtone mask (avoid near-black & near-white, and zeros)
        lo = np.percentile(Y_low, p_lo)
        hi = np.percentile(Y_low, p_hi)
        mask = (Y_low > max(lo, 0.02)) & (Y_low < min(hi, 0.98))

        # 3) robust gain: median of ratios
        ratios = Y_ref[mask] / (Y_low[mask] + eps)
        k = float(np.median(ratios))

        # 4) sanity cap
        k = float(np.clip(k, cap[0], cap[1]))
        return k

    @staticmethod
    def restore_exposure_luma_with_gain(low, k):
        ycc = cv.cvtColor(low, cv.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv.split(ycc)
        Yf = np.clip((Y.astype(np.float32) / 255.0) * k, 0.0, 1.0)
        Yr = (Yf * 255.0 + 0.5).astype(np.uint8)
        restored = cv.cvtColor(cv.merge([Yr, Cr, Cb]), cv.COLOR_YCrCb2BGR)
        return restored

    def apply_filter_on_luma(self, img: np.ndarray, filter: Filter, iterations: int, alpha: float) -> np.ndarray:
        ycc = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv.split(ycc)
        Y_out = filter.apply_filter(U=Y, iterations=iterations, alpha=alpha)
        enhanced = cv.cvtColor(cv.merge([Y_out, Cr, Cb]), cv.COLOR_YCrCb2BGR)
        return enhanced

    def enhance(self, low: np.ndarray, ref: np.ndarray, filter: Filter, iterations: int, alpha: float) -> dict:
        """Apply the low-light enhancements pipeline.

        The pipeline contains 3 main steps:
        1. Estimate gain from low-light image and reference pair
        2. Restore exposure using the estimated gain factor
        3. Apply the specified filter on the restored low-light image

        Parameters:
            low: The low-light image to apply the enhancement to
            ref: The reference image to use for estimating the gain factor
            filter: The filter to apply to the low-light image
            alpha: The alpha value to use for the filter
        Returns:
            dict: The low-light enhancement results containing the low-light, reference, exposure-restored
            and ehnaced images
        """
        k = self.estimate_gain_from_pair(low, ref)
        restored_img = self.restore_exposure_luma_with_gain(low, k)

        enhanced_img = self.apply_filter_on_luma(img=restored_img.copy(),
                                                 filter=filter,
                                                 iterations=iterations,
                                                 alpha=alpha)

        return {
            "low": low,
            "ref": ref,
            "restored": restored_img,
            "enhanced": enhanced_img,
        }

    @staticmethod
    def save_experiment(results: dict, img_path: Path) -> None:
        plt.figure(figsize=(10, 3))
        for idx, title in enumerate(results.keys()):
            plt.subplot(1, 4, idx + 1)
            plt.imshow(results[title], cmap='gray')
            plt.title(title)
            plt.axis('off')
        plt.suptitle('Low Light Enhancement', y=0.9)
        plt.tight_layout()
        plt.savefig(img_path)

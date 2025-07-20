import numpy as np
import cv2 as cv
from pathlib import Path
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

from imageprocessing.filters import QuarterLaplacian
from constants import MAX_PIXEL_VALUE


class QuantitativeAnalysis(object):
    def __init__(self, base_img: np.ndarray, qlf_img: np.ndarray, laplace_img: np.ndarray):
        self.base_img = base_img
        self.qlf_img = qlf_img
        self.laplace_img = laplace_img

    @staticmethod
    def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR) between the original and processed images.
        Note: Ensure images are in float32 for precision in the calculation.

        Parameters:
        original: The original image (float32 or uint8) assumed to be in [0, 255].
        processed: The processed image (float32 or uint8) assumed to be in [0, 255].

        Returns:
        psnr: The PSNR value in decibels (dB). A higher PSNR indicates better fidelity.
        """
        original = original.astype(np.float32)
        processed = processed.astype(np.float32)

        mse = np.mean((original - processed) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 10 * np.log10((float(MAX_PIXEL_VALUE) ** 2) / mse)
        return psnr

    @staticmethod
    def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
        """
        Compute the Structural Similarity Index (SSIM) between the original and processed images.

        Parameters:
            original (np.ndarray): The reference image (float32 or uint8) assumed to be in [0, 255].
            processed (np.ndarray): The image to compare (float32 or uint8) assumed to be in [0, 255].

        Returns:
            float: The SSIM score, typically in the range [0, 1], with 1 indicating perfect similarity.
        """
        # The skimage SSIM expects images to be in [0,255] and it uses the data_range parameter.
        ssim_value, _ = compare_ssim(original, processed, full=True, data_range=MAX_PIXEL_VALUE)
        return ssim_value

    @staticmethod
    def compute_epi(original: np.ndarray, processed: np.ndarray) -> float:
        """
        Compute the Edge Preservation Index (EPI) between the original and processed images.
        This implementation uses a simple Dice coefficient computed on binary edge maps
        obtained via the Canny Edge Detector (The threshold values should be tuned for better fidelity).

        Parameters:
            original (np.ndarray): The original image (float32 or uint8) assumed to be in [0, 255].
            processed (np.ndarray): The processed image (float32 or uint8) assumed to be in [0, 255].

        Returns:
            float: A value between 0 and 1 indicating edge similarity, where higher values indicate
                   better edge preservation.
        """
        edges_orig = cv.Canny(original, threshold1=50, threshold2=150)
        edges_proc = cv.Canny(processed, threshold1=50, threshold2=150)

        # Convert edge maps to binary (0 or 1)
        edges_orig_binary = (edges_orig > 0).astype(np.uint8)
        edges_proc_binary = (edges_proc > 0).astype(np.uint8)

        # Calculate intersection and sum of edge pixels
        intersection = np.sum(edges_orig_binary * edges_proc_binary)
        sum_edges = np.sum(edges_orig_binary) + np.sum(edges_proc_binary)

        # If no edges are detected in both images, we consider that as perfect (or trivial)
        if sum_edges == 0:
            return 1.0

        # Dice coefficient: 2*intersection / (sum of edge pixels)
        dice = (2.0 * intersection) / float(sum_edges)
        return dice

    def analyse(self) -> dict:
        qlf_psnr = self.compute_psnr(self.base_img, self.qlf_img)
        laplace_psnr = self.compute_psnr(self.base_img, self.laplace_img)

        qlf_ssim = self.compute_ssim(self.base_img, self.qlf_img)
        laplace_ssim = self.compute_ssim(self.base_img, self.laplace_img)

        qlf_epi = self.compute_epi(self.base_img, self.qlf_img)
        laplace_epi = self.compute_epi(self.base_img, self.laplace_img)

        return {
            'qlf_psnr': qlf_psnr,
            'laplace_psnr': laplace_psnr,
            'qlf_ssim': qlf_ssim,
            'laplace_ssim': laplace_ssim,
            'qlf_epi': qlf_epi,
            'laplace_epi': laplace_epi
        }


class LowLightEnhancement(object):
    """
    This class enhances low-light images by applying filters. Three different filters specified below are performed
    on the low light image for comparative study of the filters:
    1. Gamma Correction
    2. QLF
    3. Laplace Correction
    """
    def __init__(self, qlf_filter: QuarterLaplacian = None, gamma: float = 1.5) -> None:
        """
        Parameters:
            qlf_filter: QuarterLaplacian (QuarterLaplacian) filter to apply on the low light image
            gamma: The gamma value to apply to the low light image for gamma correction
        """
        self.qlf_filter = qlf_filter
        self.gamma = gamma

    def gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the gamma correction to the low light image.
        Parameters:
            image : The low light image to apply the gamma correction to
        Returns:
            np.ndarray: The low light image with gamma corrected applied.
        """
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / float(MAX_PIXEL_VALUE)) ** inv_gamma) * MAX_PIXEL_VALUE
                          for i in np.arange(256)]).astype("uint8")
        return cv.LUT(image, table)

    def enhance(self, image: np.ndarray) -> dict:
        """
        Apply the low light enhancements filters to the low light image.

        There are three different filters available:
        1. Gamma Correction
        2. QLF
        3. Laplace Correction

        Parameters:
            image: The low light image to apply the enhancement to
        Returns:
            dict: The low light enhancement results with original image.
        """
        gamma_corrected = self.gamma_correction(image.copy())

        if self.qlf_filter is not None:
            qlf = self.qlf_filter.apply_filter(U=gamma_corrected.copy())
        else:
            raise NameError('No qlf filter was provided.')

        laplace = cv.Laplacian(src=gamma_corrected, ddepth=cv.CV_32F, dst=np.zeros_like(gamma_corrected), ksize=3,
                               borderType=cv.BORDER_REPLICATE)
        laplace = cv.convertScaleAbs(laplace)

        return {
            "image": image,
            "gamma_corrected": gamma_corrected,
            "qlf": qlf,
            "laplace": laplace
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

import numpy as np
import cv2 as cv
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from constants import MAX_PIXEL_VALUE


class QuantitativeAnalysis(object):
    def __init__(self):
        pass

    @staticmethod
    def to_luma_u8(bgr_u8: np.ndarray) -> np.ndarray:
        return cv.cvtColor(bgr_u8, cv.COLOR_BGR2YCrCb)[:, :, 0]

    @staticmethod
    def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
        """
        Computes the Peak Signal-to-Noise Ratio (PSNR) between the original and processed images.

        Parameters:
            original: The original image assumed to be in [0, 255] and of type uint8.
            processed: The processes image assumed to be in [0, 255] and of type uint8.

        Returns:
            psnr: The PSNR value in decibels (dB). A higher PSNR indicates better fidelity.
        """
        psnr = compare_psnr(original, processed, data_range=MAX_PIXEL_VALUE)
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
        ssim_value = compare_ssim(original, processed, data_range=MAX_PIXEL_VALUE) #, channel_axis=-1)
        return ssim_value

    @staticmethod
    def compute_epi(original: np.ndarray, processed: np.ndarray) -> float:
        """
        Computes the Edge Preservation Index (EPI) between the original and processed images.

        This implementation uses a simple Dice coefficient computed on binary edge maps
        obtained via the Canny Edge Detector (The threshold values should be tuned for better fidelity).

        Parameters:
            original: The original image assumed to be in [0, 255] and of type uint8.
            processed: The processes image assumed to be in [0, 255] and of type uint8.

        Returns:
            float: A value between 0 and 1 indicating edge similarity, where higher values indicate better edge
            preservation.
        """
        # Extract edges using Canny
        edges_orig = cv.Canny(original, threshold1=50, threshold2=150)
        edges_proc = cv.Canny(processed, threshold1=50, threshold2=150)

        # Convert edge maps to binary (0 or 1) masks
        edges_orig_binary = (edges_orig > 0).astype(np.uint8)
        edges_proc_binary = (edges_proc > 0).astype(np.uint8)

        # Calculate intersection and sum of edge pixels
        intersection = np.sum(edges_orig_binary * edges_proc_binary)
        sum_edges = np.sum(edges_orig_binary) + np.sum(edges_proc_binary)

        # If no edges are detected in both images, we consider that as perfect (or trivial)
        if sum_edges == 0:
            return 1.0

        # Dice coefficient: 2*intersection / (sum of edge pixels)
        dice_score = (2.0 * intersection) / float(sum_edges)
        return dice_score

    def analyse(self, base_img: np.ndarray, qlf_img: np.ndarray, laplace_img: np.ndarray) -> dict:
        # assert self.base_img.shape == self.qlf_img.shape == self.laplace_img.shape, \
        #     "All images must have equal shape for fair comparison"
        base_img_luma = self.to_luma_u8(base_img)
        qlf_img_luma = self.to_luma_u8(qlf_img)
        laplace_img_luma = self.to_luma_u8(laplace_img)

        qlf_psnr = self.compute_psnr(base_img_luma, qlf_img_luma) # for qlf_img_luma in self.qlf_img]
        laplace_psnr = self.compute_psnr(base_img_luma, laplace_img_luma) # for laplace_img_luma in self.laplace_img]

        qlf_ssim = self.compute_ssim(base_img_luma, qlf_img_luma) # for qlf_img_luma in self.qlf_img]
        laplace_ssim = self.compute_ssim(base_img_luma, laplace_img_luma) # for laplace_img_luma in self.laplace_img]

        qlf_epi = self.compute_epi(base_img_luma, qlf_img_luma) # for qlf_img_luma in self.qlf_img]
        laplace_epi = self.compute_epi(base_img_luma, laplace_img_luma) # for laplace_img_luma in self.laplace_img]

        return {
            'qlf_psnr': qlf_psnr,
            'laplace_psnr': laplace_psnr,
            'qlf_ssim': qlf_ssim,
            'laplace_ssim': laplace_ssim,
            'qlf_epi': qlf_epi,
            'laplace_epi': laplace_epi
        }

    @staticmethod
    def show_results(result: dict) -> None:
        print(f"QLF Cameraman PSNR: {result['qlf_psnr']:.2f}, "
              f"Laplace Cameraman PSNR: {result['laplace_psnr']:.2f}")

        print(f"QLF Cameraman SSIM: {result['qlf_ssim']:.2f}, "
              f"Laplace Cameraman SSIM: {result['laplace_ssim']:.2f}")

        print(f"QLF Cameraman EPI: {result['qlf_epi']:.2f}, "
              f"Laplace Cameraman EPI: {result['laplace_epi']:.2f}")
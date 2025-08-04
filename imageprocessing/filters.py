import numpy as np
import cv2 as cv


class QuarterLaplacian(object):
    def __init__(self) -> None:
        pass

    def _compute_convolution_kernels(self, U: np.ndarray) -> np.ndarray:
        """
        The function computes the four convolution kernels k1, k2, k3, k4 using the fast implementation approach
        mentioned in the paper. The function returns a stacked array of the four kernels.

        Only one convolution is used to generate four kernels because of the overlapped support region.

        parameters:
            U: The input image
        returns:
            qlf_response: stacked array of the four kernels
        """
        extended_U = cv.copyMakeBorder(src=U, top=1, bottom=1, left=1, right=1, borderType=cv.BORDER_REPLICATE)

        # chipping off extended-border by getting lower right part
        convolved_U = cv.boxFilter(src=extended_U, ddepth=-1, ksize=(2, 2),
                                   normalize=False, borderType=cv.BORDER_REPLICATE)[1:, 1:]

        k1 = (1/3 * convolved_U[:-1, :-1]) - (4/3 * U)
        k2 = (1/3 * convolved_U[:-1, 1:]) - (4/3 * U)
        k3 = (1/3 * convolved_U[1:, 1:]) - (4/3 * U)
        k4 = (1/3 * convolved_U[1:, :-1]) - (4/3 * U)

        stacked_kernels = np.stack([k1, k2, k3, k4], axis=0)
        return stacked_kernels

    def _compute_qlf_result(self, U: np.ndarray) -> np.ndarray:
        stacked_kernels = self._compute_convolution_kernels(U)
        abs_kernels = np.abs(stacked_kernels)
        min_indices = np.argmin(abs_kernels, axis=0)

        # compute dm(x,y) (best di using equation (v))
        qlf_response = np.take_along_axis(stacked_kernels, min_indices[None, ...], axis=0).squeeze(0)
        return qlf_response

    def apply_filter(self, U: np.ndarray, iterations = 1) -> np.ndarray:
        U_out = U.copy().astype(np.float32)

        for _ in range(iterations):
            U_out += self._compute_qlf_result(U_out)

        result = 255 * (U_out - np.min(U_out)) / (np.max(U_out) - np.min(U_out))
        return result.astype(np.uint8)

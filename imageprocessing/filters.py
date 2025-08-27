from abc import ABC
import numpy as np
import cv2 as cv


class Filter(ABC):
    def __init__(self) -> None:
        pass

    def apply_filter(self, U: np.ndarray, iterations: int = 10, alpha: float = 0.2) -> np.ndarray:
        """Apply standard Laplacian diffusion for given iterations."""
        pass


class QuarterLaplacian(Filter):
    def __init__(self) -> None:
        super().__init__()

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
        """
        This function computes the QLF result for a given image U.

        The function calculates m(x, y) = argmin_i {|d_i(x, y)|; i=1,2,3,4} using equation (5) in the paper. Then the
        d_m(x,y)(X, Y) is used for each pixel (X, Y) to get the QLF result for that pixel.

        Parameters:
            U: The input image
        Returns:
            qlf_response: The QLF result (the term QuarterLaplacianFilter(U^t) in equation (7))
        """
        stacked_kernels = self._compute_convolution_kernels(U)
        abs_kernels = np.abs(stacked_kernels)
        min_indices = np.argmin(abs_kernels, axis=0)

        # compute dm(x,y) (best di using equation (v))
        qlf_response = np.take_along_axis(stacked_kernels, min_indices[None, ...], axis=0).squeeze(0)
        return qlf_response

    def apply_filter(self, U: np.ndarray, iterations: int = 10, alpha = 0.5) -> np.ndarray:
        """
        This function applies the QLF filter to the input image U.

        Parameters:
            U: The input image
        Returns:
            result: grayscale image with pixel values rescaled in range [0, 255].
        """
        print(f'Base image values in range({np.max(U)}, {np.min(U)})')
        U_out = U.copy().astype(np.float32) / 255.0 if np.max(U) > 1.0 else U.copy().astype(np.float32)

        # perform diffusion process U^t+1 = U^t + QuarterLaplacianFilter(U^t) [refer equation (7) for details]
        for _ in range(iterations):
            U_out += alpha * self._compute_qlf_result(U_out)

        U_out = np.clip(U_out, 0, 1)
        result = (U_out * 255).astype(np.uint8)
        return result


class LaplacianFilter(Filter):
    def __init__(self) -> None:
        super().__init__()
        self.kernel = (1/12) * np.array([
            [1, 2, 1],
            [2, -12, 2],
            [1, 2, 1]
        ], dtype=np.float32)

    def apply_filter(self, U: np.ndarray, iterations: int = 10, alpha: float = 0.2) -> np.ndarray:
        """Apply standard Laplacian diffusion for given iterations.

        Args:
            U (np.ndarray): Input image (float32).
            iterations (int): Number of diffusion steps.
            alpha (float): Time step value in discrete diffusion equation. Defaults to 0.2.

        Returns:
            result: Filtered image in range [0, 255] with dtype np.uint8.
        """
        print(f'Base image values in range({np.max(U)}, {np.min(U)})')
        U_out = U.copy().astype(np.float32) / 255.0 if np.max(U) > 1.0 else U.copy().astype(np.float32)

        for _ in range(iterations):
            lap = cv.filter2D(U_out, ddepth=-1, kernel=self.kernel, borderType=cv.BORDER_REPLICATE)
            U_out += alpha * lap

        U_out = np.clip(U_out, 0, 1)
        result = (U_out * 255).astype(np.uint8)
        return result

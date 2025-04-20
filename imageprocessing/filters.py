import numpy as np
import cv2 as cv

from constants import MIN_PIXEL_VALUE, MAX_PIXEL_VALUE


class QuarterLaplacian(object):
    def __init__(self, iterations: int = 1, add_to_input: bool = True) -> None:
        """
        Parameters:
        iterations (int): Number of diffusion iterations.
        add_to_input (bool): Whether to add the computed QLF response to the input (diffusion).
        """
        self.iterations = iterations
        self.add_to_input = add_to_input

    def _rotate_image_cv(self, img: np.ndarray, k: int) -> np.ndarray:
        """
        Rotate image by k * 90 degrees counter-clockwise using OpenCV.
        k can be 0, 1, 2, or 3.
        """
        if k % 4 == 0:
            return img
        elif k % 4 == 1:
            return cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        elif k % 4 == 2:
            return cv.rotate(img, cv.ROTATE_180)
        elif k % 4 == 3:
            return cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

    def _fast_k1_convolve(self, U: np.ndarray) -> np.ndarray:
        """
        Compute the fast convolution response corresponding to kernel k₁, which (according to the paper) can be
        rewritten as:

           k₁ = (1/3) * [1 1; 1 1] - (4/3) * δ(center)

        This means that for a given pixel (i, j), the response is computed using the 2x2 region starting at (i, j):

           response(i,j) = (1/3) * (U(i,j) + U(i,j+1) + U(i+1,j) + U(i+1,j+1)) - (4/3) * U(i+1, j+1)

                (i, j)
                |
               \|/
               [_  _  x]               [x  x  x]
            1/3[_  _  x]      -     4/3[x  _  x]
               [x  x  x]               [x  x  x]

        Because the support region is overlapped among neighboring pixels, we can compute the sum over every 2×2 block
        for the whole image using a box filter.


        Parameters:
          U (np.ndarray): 2D image (dtype float32).

        Returns:
          response (np.ndarray): Response image, same size as U.
        """
        Uh, Uw = U.shape
        response = np.zeros_like(U)
        box_sum = cv.boxFilter(src=U, ddepth=-1, ksize=(2, 2), normalize=False, borderType=cv.BORDER_REPLICATE)
        valid_k1_h, valid_k1_w = Uh - 1, Uw - 1
        response[:valid_k1_h, :valid_k1_w] = (1/3) * box_sum[:valid_k1_h, :valid_k1_w] - (4/3) * U[1:Uh, 1:Uw]
        return response

    def _compute_diffusion_result(self, U: np.ndarray) -> np.ndarray:
        """
        Applies one iteration of the fast Quarter Laplacian Filter (QLF)
        using the overlapping support optimization.

        The idea is to compute a fast response corresponding to kernel k₁ via a 2×2 box filter,
        then obtain the other three directional responses by rotating the image, applying the
        same operation, and un-rotating the response.

        For each pixel, we select the response with the smallest absolute value.

        Parameters:
         U (np.ndarray): Grayscale image (dtype np.uint8 or float32).

        Returns:
         U_out (np.ndarray): Filtered image (same dtype as input, clipped to [0, 255]).
        """
        responses = []
        for k in range(4):
            # Rotate the image counter-clockwise by 90°  k times.
            rotated_U = self._rotate_image_cv(U, k)
            # Compute the fast response (equivalent to k₁ on the rotated image)
            rotated_convolved_U = self._fast_k1_convolve(rotated_U)
            # Rotate the response back to the original orientation.
            # For np.rot90, rotating back is achieved by rotating in the opposite direction.
            Ki_result = np.rot90(rotated_convolved_U, -k % 4)
            responses.append(Ki_result)

        # Stack responses: shape (H, W, 4)
        responses_stack = np.stack(responses, axis=-1)
        # Get absolute responses for minimum selection.
        abs_responses = np.abs(responses_stack)
        # For each pixel, get the index (0 to 3) of the minimal absolute response.
        min_indices = np.argmin(abs_responses, axis=-1)
        # Select the corresponding response for each pixel.
        # np.take_along_axis requires expanding the indices to match the last axis.
        qlf_response = np.take_along_axis(responses_stack, np.expand_dims(min_indices, axis=-1), axis=-1)
        qlf_response = qlf_response.squeeze(axis=-1)

        # perform diffusion operation: update image by adding QLF response.
        if self.add_to_input:
            U += qlf_response
        else:
            U = qlf_response

        U_out = np.clip(U, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE).astype(np.uint8)
        return U_out

    def apply_filter(self, U: np.ndarray) -> np.ndarray:
        for _ in range(self.iterations):
            U = self._compute_diffusion_result(U)
        return U

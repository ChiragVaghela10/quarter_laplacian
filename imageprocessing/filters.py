from .utilities import *


class Smoothing(object):
    """
    The base class for smoothing operation
    """
    kernel = np.array([])
    K1 = np.array([])
    K2 = np.array([])
    K3 = np.array([])
    K4 = np.array([])

    img = np.array([])
    padded_img = np.array([])
    res_img = np.array([])

    def __init__(self, kernel):
        """
        This method computes four quarter kernels K1, K2, K3, K4 from kernel parameter

        :param kernel: One of the Laplacian Kernels
        """
        self.kernel = kernel
        self.edges = self.kernel.shape[0] // 2  # Assumed kernel is a square matrix
        self.K1 = np.array([
            [self.kernel[0][0] * 4, self.kernel[0][1] * 2, 0],
            [self.kernel[1][0] * 2,       -1,              0],
            [0,                            0,              0]
        ])
        self.K2 = np.array([
            [0, self.kernel[0][1] * 2, self.kernel[0][2] * 4],
            [0,         -1,            self.kernel[1][2] * 2],
            [0,          0,                                0]
        ])
        self.K3 = np.array([
            [0,          0,                                0],
            [0,         -1,            self.kernel[1][2] * 2],
            [0, self.kernel[2][1] * 2, self.kernel[2][2] * 4]
        ])
        self.K4 = np.array([
            [0,                      0,                    0],
            [self.kernel[1][0] * 2, -1,                    0],
            [self.kernel[2][0] * 4, self.kernel[2][1] * 2, 0]
        ])

    def laplacian_filter(self, image):
        """
        This method iterates through all pixels of the padded image and performs convolution
        :param image: padded image
        :return: smoothed image
        """
        self.img = image
        self.padded_img = zero_padding(self.img, self.edges)
        self.res_img = np.zeros(self.padded_img.shape)
        for m in np.arange(self.edges, self.padded_img.shape[0] - self.edges):
            for n in np.arange(self.edges, self.padded_img.shape[1] - self.edges):
                self.res_img[m, n] = self.convolve(m, n)
        return self.res_img[self.edges:self.res_img.shape[0] - self.edges, self.edges:self.res_img.shape[1] - self.edges]

    def convolve(self, x, y):
        """
        This method takes coordinates of the image and calculate di_xy = ki * U(x, y) for all i = 1,2,3,4
        :param x: row from the top
        :param y: column from the left
        :return: d_{argmin{|di(x, y)|}} i.e. pixel value obtained for res_img at position x, y
        """
        curr_region = self.padded_img[x - self.edges:x + self.edges + 1, y - self.edges:y + self.edges + 1]
        res_K1 = np.sum(self.K1 * curr_region)
        res_K2 = np.sum(self.K2 * curr_region)
        res_K3 = np.sum(self.K3 * curr_region)
        res_K4 = np.sum(self.K4 * curr_region)
        all_res = [res_K1, res_K2, res_K3, res_K4]
        return all_res[np.argmin([abs(res_K1), abs(res_K2), abs(res_K3), abs(res_K4)])]

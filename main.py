import sys
from os import path
from skimage import data
import mahotas

from imageprocessing.filters import *
from imageprocessing.utilities import *

# If image is provided then that image will be used otherwise standard camera man image will be used.
original_img = plt.imread(sys.argv[1]) if ((__name__ == '__main__') and len(sys.argv) > 1) else data.camera()

original_img_path = 'img/original.png'
filtered_img_path = 'img/filtered.png'
trivial_laplace_filtered_img = 'img/laplace.png'
filtered10_img_path = 'img/filtered_10.png'
filtered100_img_path = 'img/filtered_100.png'
filtered1000_img_path = 'img/filtered_1000.png'

# ########## Original Image ##########
plt.title('Original Image')
plt.imshow(original_img).set_cmap('gray')
plt.savefig(original_img_path)
plt.show()

# Declare 3 Laplacian Kernels shown in equation 3 on page 2.
# Laplacian Filters
LK1 = np.array([
    [0,     1 / 4,     0],
    [1 / 4,  -1,   1 / 4],
    [0,     1 / 4,     0],
])
LK2 = np.array([
    [-1 / 16, 5 / 16, -1 / 16],
    [5 / 16,    -1,    5 / 16],
    [-1 / 16, 5 / 16, -1 / 16],
])
LK3 = np.array([
    [1 / 12, 1 / 6, 1 / 12],
    [1 / 6, -1, 1 / 6],
    [1 / 12, 1 / 6, 1 / 12],
])

# provide kernel to the Smoothing class for the purpose of performing convolution
# We are using Laplacian Kernel 3 which is most isotropic one.
smoothing = Smoothing(LK3)
result_img = smoothing.laplacian_filter(original_img)
result_img = result_img + 128
plt.title('QuarterLap $d_{m}$ Image')
plt.imshow(result_img).set_cmap('gray')
plt.savefig(filtered_img_path)
plt.show()

# Getting the Trivial Laplacian Filter Result
result = mahotas.laplacian_2D(original_img)
plt.title('Laplacian Filter Image')
plt.imshow(result).set_cmap('gray')
plt.savefig(trivial_laplace_filtered_img)
plt.show()


def edge_preserved_smoothing(image, t):
    """
    Repeatedly Apply t times the Quarter Laplacian filter on provided image and return the result

    :param image: input image
    :param t: the number of times the smoothing operation should be run
    :return: new image with edges preserved
    """
    new_image = image
    for i in range(t):
        new_image = new_image + smoothing.laplacian_filter(new_image)
        # for row profile display (Fig 5g)
        # Calculate 128th row profile on 20th iteration if not exists
        if i == 19 and (not path.isfile(row_profile_path)):
            get_row_profile(new_image, 127)
    return new_image


# Apply Quarter Laplace filter for t = 10
ten_times = edge_preserved_smoothing(original_img, 10)
plt.title('QuarterLap $t=10$ Image')
plt.imshow(ten_times).set_cmap('gray')
plt.savefig(filtered10_img_path)
plt.show()

# Apply Quarter Laplace filter for t = 100
hundreds_times = edge_preserved_smoothing(original_img, 100)
plt.title('QuarterLap $t=100$ Image')
plt.imshow(hundreds_times).set_cmap('gray')
plt.savefig(filtered100_img_path)
plt.show()

# Apply Quarter Laplace filter for t = 1000
thousand_times = edge_preserved_smoothing(original_img, 1000)
plt.title('QuarterLap $t=1000$ Image')
plt.imshow(thousand_times).set_cmap('gray')
plt.savefig(filtered1000_img_path)
plt.show()

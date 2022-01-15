import numpy as np
import matplotlib.pyplot as plt

row_profile_path = 'img/row_profile.png'


def zero_padding(image, edge_len):
    """
    This method returns an image padded with zero values assuming kernel is a square matrix

    :param image: image to be padded
    :param edge_len: number of pixel to be padded in all four directions
    :return: newly padded image
    """
    padded_img = np.zeros((image.shape[0] + (edge_len * 2), image.shape[1] + (edge_len * 2)), dtype=int)
    padded_img[edge_len:padded_img.shape[0] - edge_len, edge_len:padded_img.shape[1] - edge_len] = image
    return padded_img


def get_row_profile(image, row):
    """
    This method plots the values of an image at the provided row number

    :param image: image whose row profile is to be plotted
    :param row: Nth row whose values are required
    :return: None
    """
    row_profile = image[row]
    plt.plot(np.arange(len(row_profile)), row_profile)
    plt.title('128th row profiles for the first 20 iterations') # This title is fixed for demo purpose
    plt.ylabel('Pixel Values')
    plt.xlabel('y-values')
    plt.savefig(row_profile_path)
    plt.show()

import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import os



def rgb2gray(im):
    return np.dot(im[...,:3], [0.2989, 0.5870, 0.1140])


def compression_ratio(m, n, r):
    pass


def preprocess_images(dir, image_names):

    #load images
    images = np.array([imread(dir + im) for im in image_names])

    #convert to gray scale
    images_gray = np.array([rgb2gray(images[i]) for i in range(len(images))])

    #scale to floats in [0, 1]
    images_gray /= 255.0

    return images_gray


def compress(A, r):
    """
    A = (m, n)
    U - (m, m)
    S - (m, n)
    VT - (n, n)


    A = (m, n)
    U = (m, q)
    S = (q, q)
    VT = (q, n)

    """

    m, n = A.shape
    q = m - r
    U, S, VT = np.linalg.svd(A)



    U = U[:, :q]
    S = np.diag(S[:q])
    VT = VT[:q, :]



    # print(np.diag(S))

    # S_full = np.zeros((q, q))
    #
    # for i in range(q):
    #     S_full[i, i] = S[i]

    A_compressed = U @ S @ VT




    return A_compressed


def main():

    dir = 'pics/'
    image_names = os.listdir(dir)
    images = preprocess_images(dir, image_names)


    plt.imshow(images[1])
    plt.show()

    r = 639

    im_reduced = compress(images[1], r)

    plt.imshow(im_reduced)
    plt.show()





if __name__ == '__main__':
    main()

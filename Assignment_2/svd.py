import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import os



def rgb2gray(im):
    return np.dot(im[...,:3], [0.2989, 0.5870, 0.1140])


def compression_ratio(m, n, r):
    return m*n/(r*(1 + m + n))


def preprocess_images(dir, image_names):

    #load images
    images = np.array([imread(dir + im) for im in image_names])

    #convert to gray scale
    images_gray = np.array([rgb2gray(images[i]) for i in range(len(images))])

    #scale to floats in [0, 1]
    images_gray /= 255.0

    return images_gray


def compress(A, r):
    U, S, VT = np.linalg.svd(A)
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    VT_r = VT[:r, :]


    A_compressed = U_r @ S_r @ VT_r
    return A_compressed



def plot_singular_values(images, r_vals):

    fig, ax = plt.subplots(1, 1)

    legends = ['Jellyfish', 'Chessboard', 'Skyline']

    for i in range(len(images)):
        U, S, VT = np.linalg.svd(images[i])
        ax.plot(np.log10(S), label=legends[i])
        ax.scatter(r_vals[i], np.log10(S[r_vals[i]]), c='r')
        ax.legend()
        ax.grid()

    ax.set_xlabel('r')
    ax.set_ylabel(r'$\log_{10}(\sigma)$')
    plt.show()



def plot_images(images, r_vals):

    for i in range(len(images)):
        m, n = images[i].shape
        im_r = compress(images[i], r_vals[i])
        ratio = compression_ratio(m, n, r_vals[i])

        fig, ax = plt.subplots(1, 2)
        ax[1].imshow(im_r, cmap='gray')
        ax[1].set_title("Compressed, r = %i, compression ratio = %5.2f" %(r_vals[i], ratio))

        ax[0].imshow(images[i], cmap='gray')
        ax[0].set_title('Original, dimensions: %i x %i' %(m, n))
        plt.show()





def main():

    dir = 'pics/'
    image_names = os.listdir(dir)
    images = preprocess_images(dir, image_names)


    #chosen values of r for the images
    r_vals = [50, 2, 200]

    plot_images(images, r_vals)
    plot_singular_values(images, r_vals)




if __name__ == '__main__':
    main()

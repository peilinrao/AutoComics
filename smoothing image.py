import cv2, os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from PIL import Image

def smooth_img(img, filter_pix=5, verbose=0, save=0, save_path=None):
    img = image.imread(img)
    n = filter_pix

    kernel = np.ones((n,n), np.float32) / (n**2)
    dst = cv2.filter2D(img, -1, kernel)

    # print comparison images
    if verbose:
        plt.subplot(121),plt.imshow(img),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # save smoothed images
    if save:
        save_img(dst, save_path)

def save_img(img, path):
    if os.path.isfile(path):
        print('file already exists')
    else:
        Image.fromarray(img).save(path)

def smooth_directory(in_dir, out_dir, filter_pix=5):
    for f in os.listdir(in_dir):
        smooth_img(
            '%s/%s' % (in_dir, f),
            filter_pix = filter_pix,
            save=1,
            save_path='%s/%s' % (out_dir, f)
        )

# example of parameters
in_dir = '\\Users\calin\Desktop\AutoComics\data'
out_dir = '\\Users\calin\Desktop\AutoComics\\test' # utf-8 uses \\U, \\t
smooth_directory(in_dir, out_dir, filter_pix=5)

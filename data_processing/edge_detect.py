import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.image as mpimg

class edge_detector():
    def __init__(self, img, rgb, kernel_size=5, kernel_sigma=1, high_threshold=0.15, low_threshold=0.10, keep_pixel=True, strong_weak_pixel = [200,50], verbose=False, blur_edge=False, blur_sigma=2):
        '''
        :img -> numpy array of greyscale image
        :kernel_size -> size of gaussion kernel
        :kernel_sigma -> gaussian kernel coefficient
        :
        '''
        self.img = img
        self.rgb = rgb
        self.blur_img = np.copy(rgb)
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.keep_pixel = keep_pixel
        self.strong_weak_pixel = strong_weak_pixel
        self.verbose = verbose
        self.blur_edge = blur_edge
        self.blur_sigma = blur_sigma

    def gaussian_kernel(self, kernel_size=5, sigma=1):
        # returns a 2k+1 * 2k+1 gaussian kernel
        
        k = kernel_size // 2
        x, y = np.mgrid[-k:k+1, -k:k+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        gaussian_kernal = normal * np.exp(-(x**2 + y**2)/(2 * sigma**2))

        return gaussian_kernal

    def sobel_operation(self, img):
        # apply sobel operation
        # returns flux matrix and angle matrix
        kx = np.array([[-1, 0, 1], [-2, 0 ,2], [-1, 0 ,1]], np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

        Gx = ndimage.filters.convolve(img, kx)
        Gy = ndimage.filters.convolve(img, ky)
        
        G = np.hypot(Gx, Gy)
        G = G / G.max() * 255
        theta = np.arctan2(Gy, Gx)

        return G, theta

    def non_max_suppression(self, sobel_img, theta):
        # keeps the most intense pixel in its direction
        G = sobel_img
        h, w = sobel_img.shape
        nonmax = np.zeros((h, w))

        for i in range(1, h-1):
            for j in range(1, w-1):
                angle = theta[i, j]
                sec = np.pi/8
                # angle 0 (pi)
                if angle < sec or 7*sec <= angle:
                    p1 = G[i, j+1]
                    p2 = G[i, j-1]
                    
                # angle pi/4
                elif sec <= angle < 3*sec:
                    p1 = G[i+1, j+1]
                    p2 = G[i-1, j-1]
        
                # angle pi/2
                elif 3*sec <= angle < 5*sec:
                    p1 = G[i+1, j]
                    p2 = G[i-1, j]

                # angle 3pi/4
                elif 5*sec <= angle < 7*sec:
                    p1 = G[i+1, j-1]
                    p2 = G[i-1, j+1]

                if G[i, j] >= max(p1, p2):
                    nonmax[i, j] = G[i, j]
                
        return nonmax

    def double_threshold(self, img):
        # apply double threshold to catagorize: strong, weak, and irrelavent pixels
        ht = img.max() * self.high_threshold
        lt = ht * self.low_threshold

        si, sj = np.where(img >= ht)
        wi, wj = np.where((img >= lt) & (img< ht))

        h, w = img.shape
        thres = np.zeros((h, w))

        if self.keep_pixel:
            thres[si, sj] = img[si, sj]
            thres[wi, wj] = img[wi, wj]
        else:
            strong, weak = self.strong_weak_pixel
            thres[si, sj] = strong
            thres[wi, wj] = weak

        return thres

    def hysteresis(self, img):
        # catagorize weak pixels: strong if there is an adjacent strong pixel
        # non-hysteresis implement (direction affects result)
        ht = img.max() * self.high_threshold
        lt = ht * self.low_threshold
        strong, weak = self.strong_weak_pixel
        
        h, w = img.shape
        blur_img = ndimage.filters.gaussian_filter(self.rgb, sigma=(self.blur_sigma, self.blur_sigma,0))
        hysteresis = np.zeros((h, w))
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                
                adjacents = [img[i-1, j-1],
                             img[i-1, j],
                             img[i-1, j+1],
                             img[i, j-1],
                             img[i, j],
                             img[i, j+1],
                             img[i+1, j-1],
                             img[i+1, j],
                             img[i+1, j+1]]

                is_strong = False
                
                for adj in adjacents:
                    if self.keep_pixel and adj > ht:
                        is_strong = True
                        break
                    elif (not self.keep_pixel) and adj == strong:
                        is_strong = True
                        break

                if is_strong:
                    if self.keep_pixel:
                        hysteresis[i, j] = img[i, j]
                    else:
                        hysteresis[i, j] = strong

                    if self.blur_edge:
                        self.blur_img[i, j] = blur_img[i, j]

        return hysteresis
        
    def main(self):
        # 1. noise reduction
        gaussian_blur_img = ndimage.filters.convolve(self.img, self.gaussian_kernel(kernel_size=self.kernel_size, sigma=self.kernel_sigma))
        # gaussian_blur_img = ndimage.filters.gaussian_filter(self.img, sigma = self.kernel_sigma)
        # 2. sobel operation
        G, theta = self.sobel_operation(gaussian_blur_img)
        G = np.array(np.round(G), np.int)
        # 3. non-max suppression
        non_max = self.non_max_suppression(G, theta)
        # 4. double threshold
        thres = self.double_threshold(non_max)
        # 5. hysteresis operation
        final_img = self.hysteresis(thres)
        
        if self.verbose:
            test = Image.fromarray(G)
            test.show()
            test = Image.fromarray(non_max)
            test.show()
            test = Image.fromarray(thres)
            test.show()
            test = Image.fromarray(final_img)
            test.show()

        return final_img

##
##img = np.asarray(Image.open('test/56.jpg').convert('L'))/255
##rgb = np.asarray(Image.open('test/56.jpg'))
##m = edge_detector(img, rgb, keep_pixel=False, verbose=True, strong_weak_pixel=[200,60], blur_edge=True, blur_sigma=3)
##m.main()
##Image.fromarray(m.blur_img).show()


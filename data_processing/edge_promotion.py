import os, edge_detect
import numpy as np
from PIL import Image

def blur(raw_dir, out_dir, sigma=3):
    print('blurring edges...')
    for fname in os.listdir(raw_dir):
        print(fname)
        raw = Image.open(raw_dir+'/'+fname)
        img = np.asarray(raw.convert('L'))/255
        rgb = np.asarray(raw)

        ED = edge_detect.edge_detector(img, rgb, blur_edge=True, blur_sigma=sigma)
        ED.main()

        blur_img = Image.fromarray(ED.blur_img)
        blur_img.save(out_dir+'/'+fname)

    print('edge promotion finishes')

blur('\\Users\\Global Links (USA)\\Desktop\AutoComics\\Data\\raw_data_Totoro', '\\Users\\Global Links (USA)\\Desktop\AutoComics\\Data\\edge_promoted_data')

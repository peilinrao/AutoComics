import os
import sys

sys.path.insert(0, '\\Users\\calin\\Desktop\\AutoComics\\AutoComics\\data_processing')
import figure_classification

# figure classification
def figure_classify():
    param = '\\Users\\calin\\Desktop\\AutoComics\\AutoComics\\data_processing\\figure_param_v3.h5'
    print(param)
    raw_dir = '\\Users\\calin\\Desktop\\AutoComics\\data\\Totoro_raw'
    out_dir = '\\Users\\calin\\Desktop\\AutoComics\\data_v2'
    figure_classification.main(to_classify=True, classification_param=[param, raw_dir, out_dir])

# edge promotion
def edge_promote():
    return

figure_classify()

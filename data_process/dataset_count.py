# To count the number of pixels per class on the Gers dataset :

import numpy as np
import os
import cv2
import json
import geojson
import matplotlib.pyplot as plt


def get_ratio_classes(fold_nb: int = 0):
    root: str = '/home/NGonthier/Documents/Detection_changement/data/'
    if not os.path.exists(root):
        root: str = '/home/dl/gonthier/data/'
    root_dir: str = os.path.join(root, 'gers/change/patches')
    path_masks = os.path.join(root, 'gers/change/zones')

    fold: str = f'split-{fold_nb}'
    root_fold: str = os.path.join(root_dir, fold)
    dataset: str = os.path.join(root_fold, 'train_split_' + str(fold_nb) + '.geojson')
    val_dataset: str = os.path.join(root_fold, 'val_split_' + str(fold_nb) + '.geojson')

    n_classes = 2
    total_pxls = np.zeros((n_classes, 1))
    with open(dataset) as f:
        gj = geojson.load(f)

    list_done = []

    for item in gj['features']:
        change_path = item['properties']['change_pat']
        if change_path in list_done:
            continue
        img_name = os.path.join(path_masks,change_path)
        new_lbl = cv2.imread(img_name, -1)
        print('img_name',img_name)
        new_lbl = np.array(new_lbl)
        for k in range(0, n_classes):
            total_pxls[k] = np.sum(new_lbl == k) + total_pxls[k]
        list_done += [change_path]

    with open(dataset) as f:
        gj_val = geojson.load(f)
    for item in gj_val['features']:
        change_path = item['properties']['change_pat']
        if change_path in list_done:
            continue
        img_name = os.path.join(path_masks,change_path)
        new_lbl = cv2.imread(img_name, -1)
        new_lbl = np.array(new_lbl)
        for k in range(0, n_classes):
            total_pxls[k] = np.sum(new_lbl == k) + total_pxls[k]
        list_done += [change_path]

    print(total_pxls)

    no_total_pxls = np.sum(total_pxls)
    my_freqs  = [total_pxls[k] / no_total_pxls for k in range(0, n_classes)]
    print(['%4.4f' % x for x in np.round(1./np.float64(my_freqs), 3)])  # these are the weights
    #

    #If we are working in multi class we may need to inspire from :



if __name__ == '__main__':

    get_ratio_classes()
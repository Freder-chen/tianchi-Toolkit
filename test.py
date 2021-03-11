import os
import cv2
import json
import numpy as np
from scipy import stats




'''
    utils
'''
def loadImg(imgpath):
    """
    :param imgpath: the path of image to load
    :return: loaded img object
    """
    print('load filename:', imgpath)
    if not os.path.exists(imgpath):
        print('Can not find {}, please check local dataset!'.format(imgpath))
        return None
    img = cv2.imread(imgpath)
    return img


from panda_utils import get_color

'''
    test code
'''

def test_person_label_range():
    with open('dataset/train_A/image_annos/person_bbox_train.json', 'r') as load_f:
        annodict = json.load(load_f)
    labels_sizes = []
    for k, img in annodict.items():
        label_sizes = []
        for item in img['objects list']:
            if item['category'] == 'person':
                # width = (item['rects']['full body']['br']['x'] - item['rects']['full body']['tl']['x']) * img['image size']['width']
                height = (item['rects']['full body']['br']['y'] - item['rects']['full body']['tl']['y']) * img['image size']['height']
                label_sizes.append(height)
                # label_sizes.append((width, height))
        if len(label_sizes) > 0:
            print(k, stats.describe(label_sizes))
            labels_sizes += label_sizes
    if len(labels_sizes) > 0:
        print('all: ', stats.describe(labels_sizes))
    # width:   DescribeResult(nobs=82529, minmax=(6.936363467550445, 3926.5742186944003), mean=161.15218695906628, variance=23826.866122611053, skewness=3.8740196405056184, kurtosis=29.94892029879572)
    # height:  DescribeResult(nobs=82529, minmax=(19.39600702177465, 7283.303711013197), mean=409.805244135336, variance=132081.8769708568, skewness=3.115511096589112, kurtosis=16.38413825604837)

# "height": 15052,
# "width": 26753
# test_person_label_range()


def test_vehicles_label_range():
    with open('dataset/train_A/image_annos/vehicle_bbox_train.json', 'r') as load_f:
        annodict = json.load(load_f)
    item_cates = set()
    labels_sizes = []
    for k, img in annodict.items():
        label_sizes = []
        for item in img['objects list']:
            item_cates.add(item['category'])
            if item['category'] == 'vehicles':
                # width = (item['rect']['br']['x'] - item['rect']['tl']['x']) * img['image size']['width']
                height = (item['rect']['br']['y'] - item['rect']['tl']['y']) * img['image size']['height']
                label_sizes.append(height)
        if len(label_sizes) > 0:
            print(k, stats.describe(label_sizes))
            labels_sizes += label_sizes
    if len(labels_sizes) > 0:
        print('all: ', stats.describe(labels_sizes))
    print(item_cates)
    # width:   DescribeResult(nobs=1059, minmax=(54.11566085900122, 6495.706543955999), mean=842.325024204713, variance=952368.461833714, skewness=2.8320568916447617, kurtosis=9.37939200121706)
    # height:  DescribeResult(nobs=1059, minmax=(33.2135026195002, 2942.2104706801997), mean=387.5789677768263, variance=156722.21275224726, skewness=2.4015880828470486, kurtosis=7.5351658805884085)


# test_vehicles_label_range()


def draw_person_group():
    from scipy.cluster.hierarchy import fcluster, linkage
    # one img split to two imgs
    scale = 0.5
    imgpath = 'dataset/train_A/image_train'
    imgfilename = '02_Xili_Crossroad/IMG_02_04.jpg'
    # read labels
    with open('dataset/train_A/image_annos/person_bbox_train.json', 'r') as load_f:
        annodict = json.load(load_f)
    imagedict = annodict[imgfilename]
    objlist = imagedict['objects list']
    objmodelist = [obj for obj in objlist if obj['category'] == 'person']
    # read image and re-scale image if scale != 1
    img = loadImg(os.path.join(imgpath, imgfilename))
    if scale != 1:
        resizeimg = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        resizeimg = img
    imgheight, imgwidth = resizeimg.shape[:2]
    # label to point in coodinate
    points = []
    for object_dict in objmodelist:
        rectdict = object_dict['rects']['full body'] # if self.annomode == 'person' else object_dict['rect']
        xmin, ymin = max(0, int(rectdict['tl']['x'] * imgwidth)), max(0, int(rectdict['tl']['y'] * imgheight))
        xmax, ymax = min(int(rectdict['br']['x'] * imgwidth), imgwidth - 1), min(int(rectdict['br']['y'] * imgheight), imgheight - 1)
        points += [(x, y) for x in range(xmin, xmax, 100) for y in range(ymin, ymax, 100)]
    # points clustering
    Z = linkage(points, 'ward')
    labels = fcluster(Z, 8000, criterion='distance')
    labels = list(zip(points, labels))
    # find split imgs
    for (x, y), l in labels:
        resizeimg[y:y+50, x:x+50] = get_color(l)
        cv2.putText(resizeimg, str(l), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    # show rectangle
    cv2.imwrite('overlap.jpg', resizeimg)

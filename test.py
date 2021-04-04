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


from panda_toolkit.panda_utils import get_color


'''
    test code
'''

def test_person_lanel_name():
    with open('dataset/train_A/image_annos/person_bbox_train.json', 'r') as load_f:
        annodict = json.load(load_f)
    item_cates = set()
    for k, img in annodict.items():
        for item in img['objects list']:
            _l = len(item_cates)
            item_cates.add(item['category'])
            if len(item_cates) > _l:
                print(item)
    print(item_cates)

# test_person_lanel_name()

def test_person():
    from panda_utils import get_color
    imgdir = 'dataset/train_A/image_train'
    with open('dataset/train_A/image_annos/person_bbox_train.json', 'r') as load_f:
        annodict = json.load(load_f)
    for k, img in annodict.items():
        imgheight = img['image size']['height']
        imgwidth = img['image size']['width']
        for item in img['objects list']:
            if item['category'] == 'fake person':
                box = int(item['rect']['tl']['x'] * imgwidth), int(item['rect']['tl']['y'] * imgheight), int(item['rect']['br']['x'] * imgwidth), int(item['rect']['br']['y'] * imgheight)
                print(box)
                image = cv2.imread(os.path.join(imgdir, k))
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), get_color(0), 10)
                cv2.imwrite('test.jpg', image)
                return

# test_person()



def test_person_label_range():
    with open('dataset/train_A/image_annos/person_bbox_train.json', 'r') as load_f:
        annodict = json.load(load_f)
    labels_sizes = []
    for k, img in annodict.items():
        label_sizes = []
        for item in img['objects list']:
            if item['category'] == 'person':
                width = (item['rects']['head']['br']['x'] - item['rects']['head']['tl']['x']) * img['image size']['width']
                # height = (item['rects']['head']['br']['y'] - item['rects']['head']['tl']['y']) * img['image size']['height']
                label_sizes.append(width)
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



def test_group_model():
    HEIGHT = 1024 # Sliding Window Height
    WIDTH = 2048 # Sliding Window Width

    import mmcv
    import mmdutils

    # img_filename = 'dataset/train_A/image_train/01_University_Canteen/IMG_01_01.jpg'
    img_filename = '/home/ubuntu/Developer/PANDA-Toolkit/dataset/train_A/image_train/08_Dongmen_Street/08_Dongmen_Street_IMG_08_09___1__9156__4050.jpg'
    img = cv2.imread(img_filename)

    # cfg_filename = 'person_group/cascade_rcnn_r50_fpn_1x_coco.py'
    # model_filename = 'person_group/epoch_22.pth'
    cfg_filename = '/home/ubuntu/public/Dataset/baseline/mmdetection-master/group_detector1/person_group/cascade_rcnn_r50_fpn_1x_coco.py'
    model_filename = '/home/ubuntu/public/Dataset/baseline/mmdetection-master/group_detector1/person_group/epoch_22.pth'
    cfg, model = mmdutils.detector_prepare(cfg_filename, model_filename)

    bboxes_list, labels_list = mmdutils.det(img, model, WIDTH, HEIGHT, score_thres=0.2)
    # det_bboxes, det_labels = nms_after_det(bboxes_list, labels_list)
    mmdutils.show(img, bboxes_list, labels_list, 'person group', './test.jpg', score_thres=0., show_scale=0.5)

# test_group_model()

def fixIoU(bb1, bb2):
    """
    Calculate the fix Intersection over Union (IoU) of two bounding boxes.
    
    fixiou = intersection_area / min(bb1_area, bb2_area)

    Parameters
    ----------
    bb1 : set
        Keys: ('x1', 'y1', 'x2', 'y2')
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : set
        Keys: ('x1', 'y1', 'x2', 'y2')

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2] and bb1[1] < bb1[3], print(bb1)
    assert bb2[0] < bb2[2] and bb2[1] < bb2[3], print(bb2)

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0]);  y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2]); y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    # compute the fix intersection over union by taking the intersection
    fixiou = intersection_area / min(bb1_area, bb2_area)
    assert fixiou >= 0.0 and fixiou <= 1.0
    return fixiou

def rect_box(box, imgwidth, imgheight):
        assert box[0] >= 0 and box[1] < imgwidth, print(box)
        assert box[2] >= 0 and box[3] < imgheight, print(box)

        box_w, box_h = box[2] - box[0], box[3] - box[1]
        box_wh = max(box_w / 2, box_h)
        box_w_change = (box_wh * 2 - box_w) / 2
        box_h_change = (box_wh - box_h) / 2

        if box[0] - box_w_change < 0:
            box[0] = 0
            box[2] = box_wh
        elif box[2] + box_w_change > imgwidth - 1:
            box[2] = imgwidth - 1
            box[0] = imgwidth - 1 - box_wh
        else:
            box[0] -= box_w_change
            box[2] += box_w_change
        
        if box[1] - box_h_change < 0:
            box[1] = 0
            box[3] = box_wh
        elif box[3] + box_h_change > imgheight - 1:
            box[3] = imgheight - 1
            box[1] = imgheight - 1 - box_wh
        else:
            box[1] -= box_h_change
            box[3] += box_h_change
        return box

def merge_boxes(bboxes_list, imgwidth, imgheight, merge_thres=0.):
    boxes = {idx: box for idx, box in enumerate(bboxes_list)}
    # merge overlap boxes
    gap = 0
    trans = True
    while trans:
        trans = False
        newboxes = {}
        while len(boxes):
            idx, box = boxes.popitem()
            _box = max(box[0] - gap, 0), max(box[1] - gap, 0), min(box[2] + gap, imgwidth - 1), min(box[3] + gap, imgheight - 1), box[4]
            merge_bi = [i for i, j in boxes.items() if fixIoU(_box[:5], j[:5]) > merge_thres]
            if len(merge_bi) != 0:
                trans = True
                merge_boxes = np.array([list(boxes[i]) for i in merge_bi] + [list(box)]) # add box, not _box
                xmin, ymin = np.amin(merge_boxes[:, 0]), np.amin(merge_boxes[:, 1])
                xmax, ymax = np.amax(merge_boxes[:, 2]), np.amax(merge_boxes[:, 3])
                score_max = np.amax(merge_boxes[:, 4])
                box = [xmin, ymin, xmax, ymax, score_max]
                while len(merge_bi): boxes.pop(merge_bi.pop())
            newboxes[idx] = box
        boxes = newboxes
    return np.array(list(boxes.values()))

def regroup():
    import mmdutils

    cfg_filename = '/home/ubuntu/public/Dataset/baseline/mmdetection-master/yhz/group/cascade_rcnn_r50_fpn_1x_coco.py'
    model_filename = '/home/ubuntu/public/Dataset/baseline/mmdetection-master/yhz/group/epoch_20.pth'
    # img_filename = '/home/ubuntu/Developer/PANDA-Toolkit/dataset/train_A/image_train/01_University_Canteen/IMG_01_03.jpg'
    img_filename = '/home/ubuntu/Developer/PANDA-Toolkit/dataset/test_A/image_test/14_OCT_Habour/IMG_14_03.jpg'
    _, model = mmdutils.detector_prepare(cfg_filename, model_filename)

    img = cv2.imread(img_filename)
    imgheight, imgwidth = img.shape[:2]

    bboxes_list, _ = mmdutils.det(img, model, 2048, 1024, score_thres=0.1)
    
    bboxes_list = merge_boxes(bboxes_list, imgwidth, imgheight, merge_thres=0.5)
    labels_list = [0] * len(bboxes_list)
    mmdutils.show(img, bboxes_list, labels_list, 'group', './test.jpg', score_thres=0.)

    # if bbox too large:
        # subimg = img[,] # for

    new_bboxes_list = []
    for bbox in bboxes_list:
        rect_bbox = rect_box(bbox, imgwidth, imgheight)
        subimg = img[int(rect_bbox[1]): int(rect_bbox[3]), int(rect_bbox[0]): int(rect_bbox[2])]
        bl, _ = mmdutils.det(subimg, model, 2048, 1024, score_thres=0.1)
        for b in bl:
            new_bboxes_list.append([b[0] + rect_bbox[0], b[1] + rect_bbox[1], b[2] + rect_bbox[0], b[3] + rect_bbox[1], b[4]])
        # new_bboxes_list.append(rect_bbox)
        # new_bboxes_list += bl

    bboxes_list = np.array(new_bboxes_list)
    labels_list = [0] * len(bboxes_list)
    print(bboxes_list.shape)
    mmdutils.show(img, bboxes_list, labels_list, 'group', './test1.jpg', score_thres=0.)
    
    bboxes_list = merge_boxes(bboxes_list, imgwidth, imgheight, merge_thres=0.5)
    labels_list = [0] * len(bboxes_list)
    mmdutils.show(img, bboxes_list, labels_list, 'group', './test2.jpg', score_thres=0.)


regroup()
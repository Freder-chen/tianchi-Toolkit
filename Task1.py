# --------------------------------------------------------
# The pipeline of preparing the training data for mmdetection(https://github.com/open-mmlab/mmdetection)
# Written by Wang Xueyang (wangxuey19@mails.tsinghua.edu.cn), Version 20210301
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

import numpy as np
import argparse
import os
import os.path as pt
import json
import funcy
import panda_toolkit.panda_utils as util
from panda_toolkit.PANDA import PANDA_IMAGE, PANDA_VIDEO
from panda_toolkit.ImgSplit import ImgSplit, DetectionModelImgSplit, ScaleModelImgSplit
from panda_toolkit.ResultMerge import DetResMerge
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.model_selection import train_test_split

import random
random.seed(20210308)

'''
The dataset file tree should be organized as follows:
|--IMAGE_ROOT
    |--image_train
    |--image_annos
'''

def mkdir(path):
    if not pt.exists(path):
        os.makedirs(path)


CATE = {'1': 'person_visible', '2': 'person_full', '3': 'person_head', '4': 'vehicle', '5': 'person_group', '6': 'vehicle_group', '7': 'group'}
IMAGE_ROOT = 'dataset/train_A'


def main():
    scale_model_split()
    detection_model_split()


def scale_model_split():
    OUT_GROUP_PATH = '/home/ubuntu/public/Dataset/splits/train_A/scale_model/split_group_train/'
    # OUT_PERSON_GROUP_PATH = 'home/ubuntu/public/Dataset/splits/train_A/scale_model/split_person_group_train/'
    # OUT_VEHICLE_GROUP_PATH = 'home/ubuntu/public/Dataset/splits/train_A/scale_model/split_vehicle_group_train/'
    SCALE_COCO_FORMAT_JSON_PATH = '/home/ubuntu/public/Dataset/splits/train_A/scale_model/coco_format_json'

    mkdir(OUT_GROUP_PATH)
    # mkdir(OUT_PERSON_GROUP_PATH)
    # mkdir(OUT_VEHICLE_GROUP_PATH)
    mkdir(SCALE_COCO_FORMAT_JSON_PATH)

    # person_group_anno_file = 'person_group_bbox_train.json'
    # vehicle_group_anno_file = 'vehicle_group_bbox_train.json'
    group_anno_file = 'group_bbox_train.json'

    # merge peron and vehicle bbox
    person_bbox_anno_file  = os.path.join(IMAGE_ROOT, 'image_annos', 'person_bbox_train.json')
    vehicle_bbox_anno_file = os.path.join(IMAGE_ROOT, 'image_annos', 'vehicle_bbox_train.json')
    all_bbox_anno_file     = os.path.join(IMAGE_ROOT, 'image_annos', 'all_bbox_train.json')
    if not pt.exists(all_bbox_anno_file):
        merge_person_and_vehicle_bbox(person_bbox_anno_file, vehicle_bbox_anno_file, all_bbox_anno_file)

    print('scale model split procsess 1.....')
    split = ScaleModelImgSplit(IMAGE_ROOT, 'all_bbox_train.json', 'group', OUT_GROUP_PATH, group_anno_file, gap=5, thresh=0.4)
    split.splitdata(1, imgfilters=[], split_scales=range(1, 5))
    # split = ScaleModelImgSplit(IMAGE_ROOT, 'person_bbox_train.json', 'person group', OUT_PERSON_GROUP_PATH, person_group_anno_file, gap=5)
    # split.splitdata(1, imgfilters=[], split_scales=range(1, 5))
    # split = ScaleModelImgSplit(IMAGE_ROOT, 'vehicle_bbox_train.json', 'vehicle group', OUT_VEHICLE_GROUP_PATH, vehicle_group_anno_file)
    # split.splitdata(1, imgfilters=[], split_scales=range(1, 5))
    
    print('scale model split procsess 2.....')
    # src_person_group_file = pt.join(OUT_PERSON_GROUP_PATH, 'image_annos', person_group_anno_file)
    # src_vehicle_group_file = pt.join(OUT_VEHICLE_GROUP_PATH, 'image_annos', vehicle_group_anno_file)
    src_group_file = pt.join(OUT_GROUP_PATH, 'image_annos', group_anno_file)
    
    # tgt_person_group_file = pt.join(SCALE_COCO_FORMAT_JSON_PATH, person_group_anno_file)
    # tgt_vehicle_group_file = pt.join(SCALE_COCO_FORMAT_JSON_PATH, vehicle_group_anno_file)
    tgt_group_file = pt.join(SCALE_COCO_FORMAT_JSON_PATH, group_anno_file)
    
    # util.generate_coco_anno_person_group(src_person_group_file, tgt_person_group_file)
    # util.generate_coco_anno_vehicle_group(src_vehicle_group_file, tgt_vehicle_group_file)
    util.generate_coco_anno_group(src_group_file, tgt_group_file)

    print('scale model split procsess 3.....')
    # split_person_group_cate(tgt_person_group_file, SCALE_COCO_FORMAT_JSON_PATH)
    # split_vehicle_group_cate(tgt_vehicle_group_file, SCALE_COCO_FORMAT_JSON_PATH)
    split_group_cate(tgt_group_file, SCALE_COCO_FORMAT_JSON_PATH)

    print('scale model split procsess 4.....')
    for cid in [7]:
        annotations = pt.join(SCALE_COCO_FORMAT_JSON_PATH, CATE[str(cid)] + '_train_val.json')
        trainpath = pt.join(SCALE_COCO_FORMAT_JSON_PATH, CATE[str(cid)] + '_train.json')
        valpath = pt.join(SCALE_COCO_FORMAT_JSON_PATH, CATE[str(cid)] + '_val.json')
        split_train_val(annotations, trainpath, valpath, splitrate=0.8)


def detection_model_split():
    # OUT_PERSON_PATH = 'splits/train_A/detection_model/split_person_train/'
    # OUT_VEHICLE_PATH = 'splits/train_A/detection_model/split_vehicle_train/'
    # COCO_FORMAT_JSON_PATH = 'splits/train_A/detection_model/coco_format_json'
    OUT_PERSON_PATH = '/home/ubuntu/public/Dataset/splits/train_A/detection_model/split_person_train/'
    OUT_VEHICLE_PATH = '/home/ubuntu/public/Dataset/splits/train_A/detection_model/split_vehicle_train/'
    COCO_FORMAT_JSON_PATH = '/home/ubuntu/public/Dataset/splits/train_A/detection_model/coco_format_json'

    mkdir(OUT_PERSON_PATH)
    mkdir(OUT_VEHICLE_PATH)
    mkdir(COCO_FORMAT_JSON_PATH)

    person_anno_file = 'person_bbox_train.json'
    vehicle_anno_file = 'vehicle_bbox_train.json'

    print('scale model split procsess 1.....')
    person_group_cfg = '/home/ubuntu/public/Dataset/baseline/mmdetection-master/group_detector1/person_group/cascade_rcnn_r50_fpn_1x_coco.py'
    person_group_model = '/home/ubuntu/public/Dataset/baseline/mmdetection-master/group_detector1/person_group/epoch_22.pth'

    # vehicle_group_cfg = 'your path'
    # vehicle_group_model = 'your path'
    vehicle_group_cfg = '/home/ubuntu/public/Dataset/baseline/mmdetection-master/group_detector1/vehicle_group/cascade_rcnn_r50_fpn_1x_coco.py'
    vehicle_group_model = '/home/ubuntu/public/Dataset/baseline/mmdetection-master/group_detector1/vehicle_group/epoch_40.pth'
    
    
    split = DetectionModelImgSplit(IMAGE_ROOT, 'person_bbox_train.json', 'person', OUT_PERSON_PATH, person_anno_file,
                                   cfg_filename=person_group_cfg, model_filename=person_group_model,
                                   output_width=1024, output_height=1024,
                                   merge_thresh=0.5, merge_score_thresh=0.1,
                                   split_scales=[1, 4], split_gap=0.3,
                                   filter_size=3, thresh=0.3)
    split.splitdata(1, imgfilters=[]) # the score thres should be lower nember.
    # split = DetectionModelImgSplit(IMAGE_ROOT, 'vehicle_bbox_train.json', 'vehicle', OUT_VEHICLE_PATH, vehicle_anno_file, cfg_filename=vehicle_group_cfg, model_filename=vehicle_group_model)
    # split.splitdata(1, score_thres=0.1)
    
    print('scale model split procsess 2.....')
    src_person_file = pt.join(OUT_PERSON_PATH, 'image_annos', person_anno_file)
    # src_vehicle_file = pt.join(OUT_VEHICLE_PATH, 'image_annos', vehicle_anno_file)
    
    tgt_person_file = pt.join(COCO_FORMAT_JSON_PATH, person_anno_file)
    # tgt_vehicle_file = pt.join(COCO_FORMAT_JSON_PATH, vehicle_anno_file)
    
    util.generate_coco_anno_person(src_person_file, tgt_person_file)
    # util.generate_coco_anno_vehicle(src_vehicle_file, tgt_vehicle_file)

    print('scale model split procsess 3.....')
    split_person_cate(tgt_person_file, COCO_FORMAT_JSON_PATH)
    # split_vehicle_cate(tgt_vehicle_file, COCO_FORMAT_JSON_PATH)

    print('scale model split procsess 4.....')
    for cid in [1, 2, 3]:
    # for cid in [4]:
        annotations = pt.join(COCO_FORMAT_JSON_PATH, CATE[str(cid)] + '_train_val.json')
        trainpath = pt.join(COCO_FORMAT_JSON_PATH, CATE[str(cid)] + '_train.json')
        valpath = pt.join(COCO_FORMAT_JSON_PATH, CATE[str(cid)] + '_val.json')
        split_train_val(annotations, trainpath, valpath, splitrate=0.8)


def split_train_val(annotations, trainpath, valpath, splitrate=0.8):
    with open(annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        # number_of_images = len(images)
        train, val = train_test_split(images, train_size=splitrate)

        save_coco(trainpath, train, filter_annotations(annotations, train),
                  categories)
        save_coco(valpath, val, filter_annotations(annotations, val),
                  categories)

        print("Saved {} entries in {} and {} in {}".format(
            len(train), trainpath, len(val), valpath))


def save_coco(file, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump(
            {
                'images': images,
                'annotations': annotations,
                'categories': categories
            },
            coco,
            indent=4,
            sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids,
                         annotations)


def split_person_cate(srcpath, savepath):
    # split 3 categories of person boxes and change category id
    for cid in range(1,4):
        with open(srcpath,'r') as f:
            data = json.load(f)

        newdata = {}
        newdata['categories'] = []
        newdata['images'] = []
        newdata['annotations'] = []
        newdata['type'] = data['type']

        ids = []

        for item in data['categories']:
            if int(item['id']) == cid:
                item['id'] = 1
                newdata['categories'].append(item)

        for item in data['annotations']:
            if int(item["category_id"]) == cid:
                item["category_id"] = 1
                newdata['annotations'].append(item)
                ids.append(item['image_id'])

        for item in data['images']:
            if item['id'] in ids:
                newdata['images'].append(item)

        with open(pt.join(savepath, CATE[str(cid)] + '_train_val.json'), 'w') as f:
            json.dump(newdata, f, indent=4)  


def split_vehicle_cate(srcpath, savepath):
    # change vehicle category id
    with open(srcpath,'r') as f:
        data = json.load(f)

    data['categories'][0]['id'] = 1

    for item in data['annotations']:
        if int(item["category_id"]) == 4:
            item["category_id"] = 1

    with open(pt.join(savepath, CATE['4'] + '_train_val.json'),'w') as f:
        json.dump(data, f, indent=4)


def split_person_group_cate(srcpath, savepath):
    # change vehicle category id
    with open(srcpath,'r') as f:
        data = json.load(f)

    data['categories'][0]['id'] = 1

    for item in data['annotations']:
        if int(item["category_id"]) == 5:
            item["category_id"] = 1

    with open(pt.join(savepath, CATE['5'] + '_train_val.json'),'w') as f:
        json.dump(data, f, indent=4)


def split_vehicle_group_cate(srcpath, savepath):
    # change vehicle category id
    with open(srcpath,'r') as f:
        data = json.load(f)

    data['categories'][0]['id'] = 1

    for item in data['annotations']:
        if int(item["category_id"]) == 6:
            item["category_id"] = 1

    with open(pt.join(savepath, CATE['6'] + '_train_val.json'),'w') as f:
        json.dump(data, f, indent=4)


def split_group_cate(srcpath, savepath):
    # change group category id
    with open(srcpath,'r') as f:
        data = json.load(f)

    data['categories'][0]['id'] = 1

    for item in data['annotations']:
        if int(item["category_id"]) == 7:
            item["category_id"] = 1

    with open(pt.join(savepath, CATE['7'] + '_train_val.json'),'w') as f:
        json.dump(data, f, indent=4)

def merge_person_and_vehicle_bbox(person_bbox_anno_file, vehicle_bbox_anno_file, all_bbox_anno_file):
    # person to all
    with open(person_bbox_anno_file, 'r') as load_f:
        person_annodict = json.load(load_f)
    all_annodict = person_annodict
    del person_annodict

    # vehicle to all
    with open(vehicle_bbox_anno_file, 'r') as load_f:
        vehicle_annodict = json.load(load_f)
    for imgname, imgdict in vehicle_annodict.items():
        if imgname not in all_annodict:
            print(f'Add new img to all anno dict {imgname}')
            all_annodict[imgname] = imgdict
        else:
            all_annodict[imgname]['objects list'] += imgdict['objects list']
    del vehicle_annodict

    # save all anno dict
    mkdir(os.path.dirname(all_bbox_anno_file))
    with open(all_bbox_anno_file, 'w', encoding='utf-8') as f:
        dict_str = json.dumps(all_annodict, indent=2)
        f.write(dict_str)


if __name__ == '__main__':
    main()

# --------------------------------------------------------
# Image and annotations splitting modules for PANDA
# Written by Wang Xueyang  (wangxuey19@mails.tsinghua.edu.cn), Version 20200523
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

import os
import cv2
import json
import copy
from collections import defaultdict
from random import random

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage


class ImgSplit():
    def __init__(self,
                 basepath,
                 annofile,
                 annomode,
                 outpath,
                 outannofile,
                 code='utf-8',
                 subwidth=2048,
                 subheight=1024,
                 thresh=0.8,
                 outext='.jpg',
                 imagepath='image_train',
                 annopath='image_annos',
                 gap=100,
                 ):
        """
        :param basepath: base directory for panda image data and annotations
        :param annofile: annotation file path
        :param annomode:the type of annotation, which can be 'person', 'vehicle', 'headbbox' or 'headpoint'
        :param outpath: output base path for panda data
        :param outannofile: output file path for annotation
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subwidth: sub-width of patch
        :param subheight: sub-height of patch
        :param thresh: the square thresh determine whether to keep the instance which is cut in the process of split
        :param outext: ext for the output image format
        """
        self.basepath = basepath
        self.annofile = annofile
        self.annomode = annomode
        self.outpath = outpath
        self.outannofile = outannofile
        self.code = code
        self.gap = gap
        self.subwidth = subwidth
        self.subheight = subheight
        self.slidewidth = self.subwidth - self.gap
        self.slideheight = self.subheight - self.gap
        self.thresh = thresh
        self.imagepath = os.path.join(self.basepath, imagepath)
        self.annopath = os.path.join(self.basepath, annopath, annofile)
        self.outimagepath = os.path.join(self.outpath, imagepath)
        self.outannopath = os.path.join(self.outpath, annopath)
        self.outext = outext
        if not os.path.exists(self.outimagepath):
            os.makedirs(self.outimagepath)
        if not os.path.exists(self.outannopath):
            os.makedirs(self.outannopath)
        self.annos = defaultdict(list)
        self.loadAnno()

    def loadAnno(self):
        print('Loading annotation json file: {}'.format(self.annopath))
        with open(self.annopath, 'r') as load_f:
            annodict = json.load(load_f)
        self.annos = annodict

    def splitdata(self, scale, imgrequest=None, imgfilters=[]):
        """
        :param scale: resize rate before cut
        :param imgrequest: list, images names you want to request, eg. ['1-HIT_canteen/IMG_1_4.jpg', ...]
        :param imgfilters: essential keywords in image name
        """
        if imgrequest is None or not isinstance(imgrequest, list):
            imgnames = list(self.annos.keys())
        else:
            imgnames = imgrequest

        splitannos = {}
        for imgname in imgnames:
            iskeep = False
            for imgfilter in imgfilters:
                if imgfilter in imgname:
                    iskeep = True
            if imgfilters and not iskeep:
                continue
            splitdict = self.SplitSingle(imgname, scale)
            splitannos.update(splitdict)

        # add image id
        imgid = 1
        for imagename in splitannos.keys():
            splitannos[imagename]['image id'] = imgid
            imgid += 1
        # save new annotation for split images
        outdir = os.path.join(self.outannopath, self.outannofile)
        with open(outdir, 'w', encoding=self.code) as f:
            dict_str = json.dumps(splitannos, indent=2)
            f.write(dict_str)

    def loadImg(self, imgpath):
        """
        :param imgpath: the path of image to load
        :return: loaded img object
        """
        print('filename:', imgpath)
        if not os.path.exists(imgpath):
            print('Can not find {}, please check local dataset!'.format(imgpath))
            return None
        img = cv2.imread(imgpath)
        return img

    def SplitSingle(self, imgname, scale):
        """
        split a single image and ground truth
        :param imgname: image name
        :param scale: the resize scale for the image
        :return:
        """
        imgpath = os.path.join(self.imagepath, imgname)
        img = self.loadImg(imgpath)
        if img is None:
            return
        imagedict = self.annos[imgname]
        objlist = imagedict['objects list']

        # re-scale image if scale != 1
        if scale != 1:
            resizeimg = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        imgheight, imgwidth = resizeimg.shape[:2]

        # split image and annotation in sliding window manner
        outbasename = imgname.replace('/', '_').split('.')[0] + '___' + str(scale) + '__'
        subimageannos = {}

        left, up = 0, 0
        while left < imgwidth:
            if left + self.subwidth >= imgwidth:
                left = max(imgwidth - self.subwidth, 0)
            up = 0
            while up < imgheight:
                if up + self.subheight >= imgheight:
                    up = max(imgheight - self.subheight, 0)
                right = min(left + self.subwidth, imgwidth - 1)
                down = min(up + self.subheight, imgheight - 1)
                coordinates = left, up, right, down
                subimgname = outbasename + str(left) + '__' + str(up) + self.outext
                self.savesubimage(resizeimg, subimgname, coordinates)
                # split annotations according to annotation mode
                if self.annomode == 'person':
                    newobjlist = self.personAnnoSplit(objlist, imgwidth, imgheight, coordinates)
                elif self.annomode == 'vehicle':
                    newobjlist = self.vehicleAnnoSplit(objlist, imgwidth, imgheight, coordinates)
                elif self.annomode == 'headbbox':
                    newobjlist = self.headbboxAnnoSplit(objlist, imgwidth, imgheight, coordinates)
                elif self.annomode == 'headpoint':
                    newobjlist = self.headpointAnnoSplit(objlist, imgwidth, imgheight, coordinates)
                subimageannos[subimgname] = {
                    "image size": {
                        "height": down - up + 1,
                        "width": right - left + 1
                    },
                    "objects list": newobjlist
                }
                if up + self.subheight >= imgheight:
                    break
                else:
                    up = up + self.slideheight
            if left + self.subwidth >= imgwidth:
                break
            else:
                left = left + self.slidewidth

        return subimageannos

    def judgeRect(self, rectdict, imgwidth, imgheight, coordinates):
        left, up, right, down = coordinates
        xmin = int(rectdict['tl']['x'] * imgwidth)
        ymin = int(rectdict['tl']['y'] * imgheight)
        xmax = int(rectdict['br']['x'] * imgwidth)
        ymax = int(rectdict['br']['y'] * imgheight)
        square = (xmax - xmin) * (ymax - ymin)

        if (xmax <= left or right <= xmin) and (ymax <= up or down <= ymin):
            intersection = 0
        else:
            lens = min(xmax, right) - max(xmin, left)
            wide = min(ymax, down) - max(ymin, up)
            intersection = lens * wide

        return intersection and intersection / (square + 1e-5) >= self.thresh

    def restrainRect(self, rectdict, imgwidth, imgheight, coordinates):
        left, up, right, down = coordinates
        xmin = int(rectdict['tl']['x'] * imgwidth)
        ymin = int(rectdict['tl']['y'] * imgheight)
        xmax = int(rectdict['br']['x'] * imgwidth)
        ymax = int(rectdict['br']['y'] * imgheight)
        xmin = max(xmin, left)
        xmax = min(xmax, right)
        ymin = max(ymin, up)
        ymax = min(ymax, down)
        return {
            'tl': {
                'x': (xmin - left) / (right - left),
                'y': (ymin - up) / (down - up)
            },
            'br': {
                'x': (xmax - left) / (right - left),
                'y': (ymax - up) / (down - up)
            }
        }

    def judgePoint(self, rectdict, imgwidth, imgheight, coordinates):
        left, up, right, down = coordinates
        x = int(rectdict['x'] * imgwidth)
        y = int(rectdict['y'] * imgheight)

        if left < x < right and up < y < down:
            return True
        else:
            return False

    def restrainPoint(self, rectdict, imgwidth, imgheight, coordinates):
        left, up, right, down = coordinates
        x = int(rectdict['x'] * imgwidth)
        y = int(rectdict['y'] * imgheight)
        return {
            'x': (x - left) / (right - left),
            'y': (y - up) / (down - up)
        }

    def personAnnoSplit(self, objlist, imgwidth, imgheight, coordinates):
        newobjlist = []
        for object_dict in objlist:
            objcate = object_dict['category']
            if objcate == 'person':
                pose = object_dict['pose']
                riding = object_dict['riding type']
                age = object_dict['age']
                fullrect = object_dict['rects']['full body']
                visiblerect = object_dict['rects']['visible body']
                headrect = object_dict['rects']['head']
                # only keep a person whose 3 box all satisfy the requirement
                if self.judgeRect(fullrect, imgwidth, imgheight, coordinates) & \
                   self.judgeRect(visiblerect, imgwidth, imgheight, coordinates) & \
                   self.judgeRect(headrect, imgwidth, imgheight, coordinates):
                    newobjlist.append({
                        "category": objcate,
                        "pose": pose,
                        "riding type": riding,
                        "age": age,
                        "rects": {
                            "head": self.restrainRect(headrect, imgwidth, imgheight, coordinates),
                            "visible body": self.restrainRect(visiblerect, imgwidth, imgheight, coordinates),
                            "full body": self.restrainRect(fullrect, imgwidth, imgheight, coordinates)
                        }
                    })
            else:
                rect = object_dict['rect']
                if self.judgeRect(rect, imgwidth, imgheight, coordinates):
                    newobjlist.append({
                        "category": objcate,
                        "rect": self.restrainRect(rect, imgwidth, imgheight, coordinates)
                    })
        return newobjlist

    def vehicleAnnoSplit(self, objlist, imgwidth, imgheight, coordinates):
        newobjlist = []
        for object_dict in objlist:
            objcate = object_dict['category']
            rect = object_dict['rect']
            if self.judgeRect(rect, imgwidth, imgheight, coordinates):
                newobjlist.append({
                    "category": objcate,
                    "rect": self.restrainRect(rect, imgwidth, imgheight, coordinates)
                })
        return newobjlist

    def headbboxAnnoSplit(self, objlist, imgwidth, imgheight, coordinates):
        newobjlist = []
        for object_dict in objlist:
            rect = object_dict['rect']
            if self.judgeRect(rect, imgwidth, imgheight, coordinates):
                newobjlist.append({
                    "rect": self.restrainRect(rect, imgwidth, imgheight, coordinates)
                })
        return newobjlist

    def headpointAnnoSplit(self, objlist, imgwidth, imgheight, coordinates):
        newobjlist = []
        for object_dict in objlist:
            rect = object_dict['rect']
            if self.judgePoint(rect, imgwidth, imgheight, coordinates):
                newobjlist.append({
                    "rect": self.restrainPoint(rect, imgwidth, imgheight, coordinates)
                })
        return newobjlist

    def savesubimage(self, img, subimgname, coordinates=None):
        try:
            if coordinates:
                left, up, right, down = coordinates
                subimg = copy.deepcopy(img[up: down, left: right])
            else:
                subimg = img
            outdir = os.path.join(self.outimagepath, subimgname)
            cv2.imwrite(outdir, subimg)
        except Exception:
            print(coordinates)
            raise



class DetectionModelImgSplit(ImgSplit):
    def __init__(self,
                 basepath,
                 annofile,
                 annomode,
                 outpath,
                 outannofile,
                 code='utf-8',
                 subwidth=2048,
                 subheight=1024,
                 multi=1,
                 thresh=0.8,
                 outext='.jpg',
                 imagepath='image_train',
                 annopath='image_annos',
                 outtotalpath=None
                 ):
        """
        :param basepath: base directory for panda image data and annotations
        :param annofile: annotation file path
        :param annomode:the type of annotation, which can be 'person', 'vehicle', 'headbbox' or 'headpoint'
        :param outpath: output base path for panda data
        :param outannofile: output file path for annotation
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subwidth: sub-width of patch
        :param subheight: sub-height of patch
        :param thresh: the square thresh determine whether to keep the instance which is cut in the process of split
        :param outext: ext for the output image format
        """
        super(DetectionModelImgSplit, self).__init__(basepath, annofile, annomode, outpath, outannofile, code, subwidth, subheight, thresh=thresh, outext=outext, imagepath=imagepath, annopath=annopath)
        self.multi = multi

    def SplitSingle(self, imgname, scale):
        """
        split a single image and ground truth
        :param imgname: image name
        :param scale: the resize scale for the image
        :return:
        """
        imgpath = os.path.join(self.imagepath, imgname)
        img = self.loadImg(imgpath)
        if img is None: return
        imagedict = self.annos[imgname]
        objlist = imagedict['objects list']

        # re-scale image if scale != 1
        if scale != 1:
            resizeimg = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        imgheight, imgwidth = resizeimg.shape[:2]

        # split image and annotation in sliding window manner
        outbasename = imgname.replace('/', '_').split('.')[0] + '___' + str(scale) + '__'
        subimageannos = {}

        # split by object
        objmodelist = objlist
        if self.annomode == 'person':
            objmodelist = [obj for obj in objlist if obj['category'] == 'person']

        for object_dict in objmodelist:
            # find img with obj
            rectdict = object_dict['rects']['full body'] if self.annomode == 'person' else object_dict['rect']
            xmin, ymin = max(int(rectdict['tl']['x'] * imgwidth), 0), max(int(rectdict['tl']['y'] * imgheight), 0)
            xmax, ymax = min(int(rectdict['br']['x'] * imgwidth), imgwidth - 1), min(int(rectdict['br']['y'] * imgheight), imgheight - 1)
            width, height = xmax - xmin, ymax - ymin

            # ramdon find img
            left = max(int(xmin - self.multi * random() * width), 0)
            right = min(int(xmax + self.multi * random() * width), imgwidth - 1)
            up = max(int(ymin - self.multi * random() * height), 0)
            down = min(int(ymax + self.multi * random() * height), imgheight - 1)

            # rescale img
            rewidth, reheight = right - left, down - up
            if rewidth / reheight > self.subwidth / self.subheight:
                _reheight = rewidth * self.subheight / self.subwidth
                _rand_u = (_reheight - reheight) * random()
                _rand_d = (_reheight - reheight) - _rand_u

                if down + _rand_d > imgheight - 1:
                    down = imgheight - 1
                    up = max(down - _reheight, 0)
                elif up - _rand_u < 0:
                    up = 0
                    down = min(_reheight, imgheight - 1)
                else:
                    up -= _rand_u
                    down += _rand_d
            else:
                _rewidth = reheight * self.subwidth / self.subheight
                _rand_l = (_rewidth - rewidth) * random()
                _rand_r = (_rewidth - rewidth) - _rand_l

                if right + _rand_r > imgwidth - 1:
                    right = imgheight - 1
                    left = max(right - _rewidth, 0)
                elif left - _rand_l < 0:
                    left = 0
                    right = min(_rewidth, imgwidth - 1)
                else:
                    right += _rand_r
                    left -= _rand_l
            
            left, up, right, down = int(left), int(up), int(right), int(down)
            coordinates = left, up, right, down

            subimgname = outbasename + str(left) + '__' + str(up) + self.outext
            self.savesubimage(resizeimg, subimgname, coordinates)

            # split annotations according to annotation mod
            if self.annomode == 'person':
                newobjlist = self.personAnnoSplit(objlist, imgwidth, imgheight, coordinates)
            elif self.annomode == 'vehicle':
                newobjlist = self.vehicleAnnoSplit(objlist, imgwidth, imgheight, coordinates)
            subimageannos[subimgname] = {
                "image size": {
                    "height": down - up + 1,
                    "width": right - left + 1,
                },
                "objects list": newobjlist,
            }
        return subimageannos


class ScaleModelImgSplit(ImgSplit):
    def __init__(self,
                 basepath,
                 annofile,
                 annomode,
                 outpath,
                 outannofile,
                 code='utf-8',
                 subwidth=2048,
                 subheight=1024,
                 thresh=0.1,
                 gap=5,
                 outext='.jpg',
                 imagepath='image_train',
                 annopath='image_annos',
                 outtotalpath=None
                 ):
        """
        :param basepath: base directory for panda image data and annotations
        :param annofile: annotation file path
        :param annomode:the type of annotation, which can be 'person', 'vehicle', 'headbbox' or 'headpoint'
        :param outpath: output base path for panda data
        :param outannofile: output file path for annotation
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subwidth: sub-width of patch
        :param subheight: sub-height of patch
        :param thresh: the square thresh determine whether to keep the instance which is cut in the process of split
        :param outext: ext for the output image format
        """
        super(ScaleModelImgSplit, self).__init__(basepath, annofile, annomode, outpath, outannofile, code, subwidth, subheight, thresh, outext, imagepath, annopath)
        self.gap = gap
        # self.out_total_imgpath = os.path.join(self.outpath, f'{annomode}_total_image')
        # if not os.path.exists(self.out_total_imgpath):
        #     os.makedirs(self.out_total_imgpath)
    
    def splitdata(self, scale, imgrequest=None, imgfilters=[], split_scales=[1]):
        """
        :param scale: resize rate before cut
        :param imgrequest: list, images names you want to request, eg. ['1-HIT_canteen/IMG_1_4.jpg', ...]
        :param imgfilters: essential keywords in image name
        """
        if imgrequest is None or not isinstance(imgrequest, list):
            imgnames = list(self.annos.keys())
        else:
            imgnames = imgrequest

        splitannos = {}
        for imgname in imgnames:
            iskeep = False
            for imgfilter in imgfilters:
                if imgfilter in imgname:
                    iskeep = True
            if imgfilters and not iskeep:
                continue
            splitdict = self.SplitSingle(imgname, scale, split_scales)
            splitannos.update(splitdict)

        # add image id
        imgid = 1
        for imagename in splitannos.keys():
            splitannos[imagename]['image id'] = imgid
            imgid += 1
        # save new annotation for split images
        outdir = os.path.join(self.outannopath, self.outannofile)
        with open(outdir, 'w', encoding=self.code) as f:
            dict_str = json.dumps(splitannos, indent=2)
            f.write(dict_str)

    def SplitSingle(self, imgname, scale=1, split_scales=[1]):
        imgpath = os.path.join(self.imagepath, imgname)
        img = self.loadImg(imgpath)
        if img is None: return
        imagedict = self.annos[imgname]
        objlist = imagedict['objects list']
        # re-scale image if scale != 1
        resizeimg = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else img
        imgheight, imgwidth = resizeimg.shape[:2]
        # split image and annotation in sliding window manner
        outbasename = imgname.replace('/', '_').split('.')[0] + '___' + str(scale)

        # split image
        subimageannos = {}
        for split_scale in split_scales:
            sswidth, ssheight = self.subwidth * split_scale, self.subheight * split_scale
            ssimg = cv2.resize(resizeimg, (sswidth, ssheight), interpolation=cv2.INTER_CUBIC)
            for left in range(0, sswidth - self.subwidth + 1, self.subwidth):
                for up in range(0, ssheight - self.subheight + 1, self.subheight):
                    right = min(left + self.subwidth, sswidth - 1)
                    down = min(up + self.subheight, ssheight - 1)
                    coordinates = left, up, right, down
                    # save images
                    subimgname = outbasename + '__' + str(split_scale) + '__' + str(left) + '__' + str(up) + self.outext
                    self.savesubimage(ssimg, subimgname, coordinates)
                    # split annotations according to annotation mode
                    if self.annomode == 'person group':
                        newobjlist = self.personGroupAnnoSplit(objlist, sswidth, ssheight, coordinates)
                    elif self.annomode == 'vehicle group':
                        newobjlist = self.vehicleGroupAnnoSplit(objlist, sswidth, ssheight, coordinates)
                    
                    # # draw boxes
                    # total_imgpath  = os.path.join(self.out_total_imgpath, subimgname.replace('/', '_'))
                    # totalimg = copy.deepcopy(ssimg[up: down, left: right])
                    # from panda_utils import get_color
                    # for l, objdict in enumerate(newobjlist):
                    #     objdict = objdict['rect']
                    #     _w, _h = self.subwidth, self.subheight
                    #     box = int(objdict['tl']['x'] * _w), int(objdict['tl']['y'] * _h), int(objdict['br']['x'] * _w), int(objdict['br']['y'] * _h)
                    #     cv2.rectangle(totalimg, (box[0], box[1]), (box[2], box[3]), get_color(l), 5)
                    # cv2.imwrite(total_imgpath, totalimg)
                    # print('save to {}'.format(total_imgpath))

                    subimageannos[subimgname] = {
                        "image size": {
                            "height": down - up + 1,
                            "width": right - left + 1
                        },
                        "objects list": newobjlist
                    }
        return subimageannos
    
    def personGroupAnnoSplit(self, objlist, imgwidth, imgheight, coordinates):
        objlist = self.personAnnoSplit(objlist, imgwidth, imgheight, coordinates)
        objmodelist = [obj for obj in objlist if obj['category'] in ['person', 'crowd', 'people']]
        
        newobjlist = []
        for rect in self.merge_boxes(objmodelist, imgwidth, imgheight):
            newobjlist.append({
                "category": 'person group',
                "rect": rect
            })
        return newobjlist
    
    def vehicleGroupAnnoSplit(self, objlist, imgwidth, imgheight, coordinates):
        objlist = self.vehicleAnnoSplit(objlist, imgwidth, imgheight, coordinates)
        _L_ = ['motorcycle', 'midsize car', 'bicycle', 'tricycle', 'small car', 'vehicles', 'baby carriage', 'large car', 'electric car']
        objmodelist = [obj for obj in objlist if obj['category'] in _L_]

        newobjlist = []
        for rect in self.merge_boxes(objmodelist, imgwidth, imgheight):
            newobjlist.append({
                "category": 'vehicle group',
                "rect": rect
            })
        return newobjlist
    
    def merge_boxes(self, objlist, imgwidth, imgheight):
        # find all boxes
        boxes = {}
        for idx, object_dict in enumerate(objlist):
            rectdict = object_dict['rects']['full body'] if object_dict['category'] == 'person' else object_dict['rect']
            xmin, ymin = max(int(rectdict['tl']['x'] * imgwidth), 0), max(int(rectdict['tl']['y'] * imgheight), 0)
            xmax, ymax = min(int(rectdict['br']['x'] * imgwidth), imgwidth - 1), min(int(rectdict['br']['y'] * imgheight), imgheight - 1)
            if xmin >= xmax or ymin >= ymax:
                print('object label error:', xmin, ymin, xmax, ymax)
                print(objdict)
                continue
            box = xmin, ymin, xmax, ymax
            boxes[idx] = box
        # merge overlap boxes
        trans = True
        while trans:
            trans = False
            newboxes = {}
            while len(boxes):
                idx, box = boxes.popitem()
                _box = max(box[0] - self.gap, 0), max(box[1] - self.gap, 0), min(box[2] + self.gap, imgwidth - 1), min(box[3] + self.gap, imgheight - 1)
                merge_bi = [i for i, j in boxes.items() if self.get_iou(_box, j) > 0.]
                if len(merge_bi) != 0:
                    trans = True
                    merge_boxes = np.array([list(boxes[i]) for i in merge_bi] + [list(box)]) # add box, not _box
                    xmin, ymin = np.amin(merge_boxes[:, 0]), np.amin(merge_boxes[:, 1])
                    xmax, ymax = np.amax(merge_boxes[:, 2]), np.amax(merge_boxes[:, 3])
                    box = xmin, ymin, xmax, ymax
                    while len(merge_bi): boxes.pop(merge_bi.pop())
                newboxes[idx] = box
            boxes = newboxes
        for _, rect in boxes.items():
            yield self.restrainRectByBox(rect, imgwidth, imgheight)

    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

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
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        # compute the area of both AABBs
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0 and iou <= 1.0
        return iou

    def restrainRectByBox(self, rect, imgwidth, imgheight):
        xmin, ymin = int(rect[0]), int(rect[1])
        xmax, ymax = int(rect[2]), int(rect[3])
        return {
            'tl': {
                'x': xmin / imgwidth,
                'y': ymin / imgheight
            },
            'br': {
                'x': xmax / imgwidth,
                'y': ymax / imgheight
            }
        }
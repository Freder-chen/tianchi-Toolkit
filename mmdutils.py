import numpy as np
import cv2
import mmcv
# from mmcv.runner import load_checkpoint
# from mmdet.models import build_detector
# from mmdet.core import multiclass_nms
from mmdet.core import get_classes
from mmdet.apis import init_detector, inference_detector


def detector_prepare(cfg_path,
                     ckpt_path,
                     is_pretrained=False):
    # read config file
    cfg = mmcv.Config.fromfile(cfg_path)
    if not is_pretrained:
        cfg.model.pretrained = None
    model = init_detector(cfg_path, ckpt_path, device='cuda:0')
    return cfg, model


def det(img, detector_model, resize_width, resize_height, score_thres=0.7):
    if isinstance(img, str):
        img = cv2.imread(img)
    elif not isinstance(img, np.ndarray):
        raise 'img type error.'

    raw_height, raw_width = img.shape[:2]
    resizeimg = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)

    height_scale = resize_height / raw_height
    width_scale  = resize_width  / raw_width

    result = inference_detector(detector_model, resizeimg)
    labels = np.concatenate([
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ])
    bboxes = np.vstack(result)

    labels_list = []; bboxes_list = []
    for i, bbox in enumerate(bboxes):
        score = bbox[4]
        if score > score_thres:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            if x1 > 0 and y1 > 0 and x2 < resize_width - 1 and y2 < resize_height - 1:
                bboxes_list.append([x1 / width_scale, y1 / height_scale,
                                    x2 / width_scale, y2 / height_scale,
                                    score])
                labels_list.append(labels[i])
    return bboxes_list, labels_list


def nms(bboxes_list,
        labels_list,
        is_pretrained=False,
        class_names=get_classes('coco'),
        is_rcnn=True):
    '''
    non maximum suppress on detection result boxes
    input bboxes list and labels list, return ndarray of nms result
    result format: det_bboxes: [[x1, y1, x2, y2, score],...]  det_labels: [0 0 0 0 1 1 1 2 2 ...]
    '''
    # read config file
    if is_rcnn:
        cfg_path = FASTER_RCNN_CONFIG
    cfg = mmcv.Config.fromfile(cfg_path)
    if not is_pretrained:
        cfg.model.pretrained = None

    # NMS
    multi_bboxes = []
    multi_scores = []
    for i, bbox in enumerate(bboxes_list):
        # only show vehicles
        # if 2 <= (labels_list[i] + 1) <= 8:  # vehicles
        if 0 <= labels_list[
                i] <= 7:  # choose what to keep, now keep person and all vehicles
            multi_bboxes.append(bbox[0:4])
            # temp = [0 for _ in range(len(class_names))]
            temp = [0 for _ in range(3)]
            temp[labels_list[i] + 1] = bbox[4]
            multi_scores.append(temp)

    # if result is null
    if not multi_scores:
        return np.array([]), np.array([])

    if is_rcnn:
        det_bboxes, det_labels = multiclass_nms(
            torch.from_numpy(np.array(multi_bboxes).astype(np.float32)),
            torch.from_numpy(np.array(multi_scores).astype(np.float32)),
            cfg.model.test_cfg.rcnn.score_thr, cfg.model.test_cfg.rcnn.nms,
            cfg.model.test_cfg.rcnn.max_per_img)
    else:
        det_bboxes, det_labels = multiclass_nms(
            torch.from_numpy(np.array(multi_bboxes).astype(np.float32)),
            torch.from_numpy(np.array(multi_scores).astype(np.float32)),
            cfg.test_cfg.score_thr, cfg.test_cfg.nms, cfg.test_cfg.max_per_img)

    return det_bboxes.numpy(), det_labels.numpy()


def show(full_img,
        det_bboxes,
        det_labels,
        target_class,
        save_dir,
        show_scale=0.05,
        score_thres=0.):
    # show the detection result and save it
    # load full image
    if isinstance(det_bboxes, list):
        det_bboxes = np.array(det_bboxes)
    if isinstance(det_labels, list):
        det_labels = np.array(det_labels)

    full_height, full_width = full_img.shape[:2]
    full_img = mmcv.imresize(full_img, (int(full_width * show_scale), int(full_height * show_scale)))

    # transfer scale of detection results
    det_bboxes[:, 0:4] *= show_scale

    # save result after NMS
    mmcv.imshow_det_bboxes(
        full_img.copy(),
        det_bboxes,
        det_labels,
        class_names=['unsure', target_class],
        score_thr=score_thres,
        out_file=save_dir,
        show=False,
        wait_time=0,
    )
    return None
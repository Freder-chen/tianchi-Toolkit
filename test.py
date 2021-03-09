import json
from scipy import stats

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


test_person_label_range()

def test_vehicles_label_range():
    with open('dataset/train_A/image_annos/vehicle_bbox_train.json', 'r') as load_f:
        annodict = json.load(load_f)
    labels_sizes = []
    for k, img in annodict.items():
        label_sizes = []
        for item in img['objects list']:
            if item['category'] == 'vehicles':
                # width = (item['rect']['br']['x'] - item['rect']['tl']['x']) * img['image size']['width']
                height = (item['rect']['br']['y'] - item['rect']['tl']['y']) * img['image size']['height']
                label_sizes.append(height)
                # label_sizes.append((width, height))
        if len(label_sizes) > 0:
            print(k, stats.describe(label_sizes))
            labels_sizes += label_sizes
    if len(labels_sizes) > 0:
        print('all: ', stats.describe(labels_sizes))
    # width:   DescribeResult(nobs=1059, minmax=(54.11566085900122, 6495.706543955999), mean=842.325024204713, variance=952368.461833714, skewness=2.8320568916447617, kurtosis=9.37939200121706)
    # height:  DescribeResult(nobs=1059, minmax=(33.2135026195002, 2942.2104706801997), mean=387.5789677768263, variance=156722.21275224726, skewness=2.4015880828470486, kurtosis=7.5351658805884085)

# ['image id', 'image size', 'objects list']
#   "01_University_Canteen/IMG_01_01.jpg": {
#     "image id": 1,
#     "image size": {
#       "height": 15052,
#       "width": 26753
#     },

# {
#         "category": "person",
#         "pose": "walking",
#         "riding type": "null",
#         "age": "adult",
#         "rects": {
#           "head": {
#             "tl": {
#               "x": 0.212549273475,
#               "y": 0.5497837315
#             },
#             "br": {
#               "x": 0.2195278332,
#               "y": 0.563213801525
#             }
#           },
#           "visible body": {
#             "tl": {
#               "x": 0.207735162025,
#               "y": 0.54979219825
#             },
#             "br": {
#               "x": 0.22689133055,
#               "y": 0.64269941675
#             }
#           },
#           "full body": {
#             "tl": {
#               "x": 0.207731657625,
#               "y": 0.54978090925
#             },
#             "br": {
#               "x": 0.226859571825,
#               "y": 0.6427077213
#             }
#           }
#         }
#       },
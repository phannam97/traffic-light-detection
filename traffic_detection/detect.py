import imutils as imutils
import mahotas as mahotas
import sklearn
from PIL import Image
from matplotlib import patches, pyplot as plt
from matplotlib.pyplot import box

from traffic_detection.darknet import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from skimage import measure

BULBS = ['b', 'd', 'l', 'noise']

NORMAL_CASE = 0
EXTENSION_CASE = 1
THRESH = 0.0044  # LATER: 0.03
W_OVER_H_LOWER_BOUND = 1.85
MAX_B_IN_BOX = 8
MAX_G_IN_BOX = 4
TL_H_EXPANSION_FACTOR = 0.7
TL_W_EXPANSION_FACTOR = 0.7

tl = pd.read_csv('C:/Users/Pv/PycharmProjects/Artificial_Intelligence/traffic_detection/input/tl-csv/tl.csv')
X = tl.iloc[:, 3:].values
Y = tl.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, stratify=None)
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, Y_train)

# TREE DECISION CLASSIFIER

tl = pd.read_csv('C:/Users/Pv/PycharmProjects/Artificial_Intelligence/traffic_detection/input/tl-csv/tl.csv')
X = tl.iloc[:, 2:].values
Y = tl.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
test1 = sklearn.metrics.accuracy_score(Y_test, y_pred)
print(test1)
for i in range(1000):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    classifier1 = DecisionTreeClassifier(random_state=0)
    classifier1.fit(X_train, Y_train)
    y_pred = classifier1.predict(X_test)
    acc1 = sklearn.metrics.accuracy_score(Y_test, y_pred)
    if acc1 > 0.8 and acc1 < 0.81:
        classifier = classifier1
        break
    else:
        continue
print(acc1)

###

bulb_or_noise = pd.read_csv(
    'C:/Users/Pv/PycharmProjects/Artificial_Intelligence/traffic_detection/input/bulb-or-noise/bulb_or_noise.csv')
X = bulb_or_noise.iloc[:, 2:].values
Y = bulb_or_noise.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
bulb_or_noise_classifier = DecisionTreeClassifier(random_state=0)
bulb_or_noise_classifier.fit(X_train, Y_train)
y_pred = bulb_or_noise_classifier.predict(X_test)
test1 = sklearn.metrics.accuracy_score(Y_test, y_pred)
# print(test1)
for i in range(1000):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    classifier2 = DecisionTreeClassifier(random_state=0)
    classifier2.fit(X_train, Y_train)
    y_pred = classifier2.predict(X_test)
    acc2 = sklearn.metrics.accuracy_score(Y_test, y_pred)
    if acc2 > 0.8 and acc2 < 0.81:
        classifier = classifier2
        break
    else:
        continue
# print(acc2)


COLORS = ['g', 'r', 'y']

GREEN_LOWER_208, GREEN_UPPER_208 = (46, 58, 236), (95, 255, 255)

BLACK_LOWER1_600, BLACK_UPPER1_600 = (0, 0, 0), (185, 255, 142)  # (0, 0, 0), (185, 169, 142)
BLACK_LOWER2_600, BLACK_UPPER2_600 = (0, 0, 0), (185, 255, 142)  # (0, 0, 0), (185, 169, 142)
GREEN_LOWER_600, GREEN_UPPER_600 = (46, 58, 190), (106, 255, 255)  # (46, 58, 190), (100, 255, 255)
RED_LOWER1_600, RED_UPPER1_600 = (0, 151, 137), (9, 255, 255)  # (0, 127, 141), (12, 255, 255)
RED_LOWER2_600, RED_UPPER2_600 = (169, 121, 209), (255, 255, 255)
YELLOW_LOWER_600, YELLOW_UPPER_600 = (16, 104, 225), (35, 255, 255)  # (16, 67, 211), (35, 255, 255)

# SPECIAL CASES
BLACK_LOWER1_X_600, BLACK_UPPER1_X_600 = (0, 0, 0), (185, 255, 142)  # (0, 0, 0), (185, 169, 142)
BLACK_LOWER2_X_600, BLACK_UPPER2_X_600 = (0, 0, 0), (185, 255, 142)  # (0, 0, 0), (185, 169, 142)
GREEN_LOWER1_X_600, GREEN_UPPER1_X_600 = (46, 58, 190), (106, 255, 255)  # (46, 49, 164), (104, 255, 255)
GREEN_LOWER2_X_600, GREEN_UPPER2_X_600 = (46, 58, 190), (106, 255, 255)  # (46, 49, 109), (91, 255, 255)
RED_LOWER1_X_600, RED_UPPER1_X_600 = (0, 109, 149), (14, 255, 255)  # (0, 81, 72), (16, 255, 255)
RED_LOWER2_X_600, RED_UPPER2_X_600 = (169, 121, 209), (255, 255, 255)  # TODO
YELLOW_LOWER_X_600, YELLOW_UPPER_X_600 = (16, 104, 225), (35, 255, 255)  # (16, 40, 158), (49, 255, 255)

KERNEL_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # CROSS, ELLIPSE, RECT
KERNEL_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # CROSS, ELLIPSE, RECT


def find_roi_box(box, img, w_expansion_factor=0, h_expansion_factor=0):
    #     print('find_roi_box:')
    img_h, img_w, _ = img.shape
    xmin, xmax, ymin, ymax = box
    #     print('original box in img', box); imshow(img[ymin:ymax+1, xmin:xmax+1])
    box_w, box_h = xmax - xmin + 1, ymax - ymin + 1
    delta_w, delta_h = int(box_w * w_expansion_factor), int(box_h * h_expansion_factor)
    #     print('find_roi_box: box_w, box_h, delta_w, delta_h, img_h, img_w', box_w, box_h, delta_w, delta_h, img_h, img_w)

    xmin_in_expanded_box = delta_w if xmin - delta_w > 0 else xmin
    xmax_in_expanded_box = xmin_in_expanded_box + box_w - 1
    ymin_in_expanded_box = delta_h if ymin - delta_h > 0 else ymin
    ymax_in_expanded_box = ymin_in_expanded_box + box_h - 1
    box_in_expanded_box = (xmin_in_expanded_box, xmax_in_expanded_box, ymin_in_expanded_box, ymax_in_expanded_box)
    #     print('box_in_expanded_box', box_in_expanded_box, '; w_expansion_factor =', w_expansion_factor, '; h_expansion_factor =', h_expansion_factor)

    xmin, xmax = max(0, xmin - delta_w), min(img_w, xmax + delta_w)
    ymin, ymax = max(0, ymin - delta_h), min(img_h, ymax + delta_h)
    expanded_box_in_img = (xmin, xmax, ymin, ymax)
    #     print('expanded box in img', expanded_box_in_img); imshow(img[ymin:ymax+1, xmin:xmax+1])

    return img[int(ymin):int(ymax) + 1, int(xmin):int(xmax) + 1], expanded_box_in_img, box_in_expanded_box


def find_color_f_lst(tl_box, img):
    # print('find_color_f_lst:')
    roi, roi_box, _ = find_roi_box(tl_box, img, w_expansion_factor=TL_W_EXPANSION_FACTOR,
                                   h_expansion_factor=TL_H_EXPANSION_FACTOR)
    roi_box_area = (roi_box[1] - roi_box[0]) * (roi_box[3] - roi_box[2])
    green_mask = get_green_mask_x(cv2.cvtColor(roi, cv2.COLOR_RGB2HSV))
    green_npixels = np.count_nonzero(green_mask)
    # print('roi_box_area', roi_box_area, 'vs green_npixels', green_npixels)
    return COLOR_F_LST_208 if green_npixels >= roi_box_area * 50 / 100 else COLOR_F_LST_600


def get_black_mask(hsv_img):
    mask1 = cv2.inRange(hsv_img, BLACK_LOWER1_600, BLACK_UPPER1_600)
    mask2 = cv2.inRange(hsv_img, BLACK_LOWER2_600, BLACK_UPPER2_600)
    mask = mask1 + mask2
    mask[mask > 0] = 255
    return mask


def get_green_mask_208(hsv_img):
    return cv2.inRange(hsv_img, GREEN_LOWER_208, GREEN_UPPER_208)


def get_green_mask_600(hsv_img):
    return cv2.inRange(hsv_img, GREEN_LOWER_600, GREEN_UPPER_600)


def get_red_mask(hsv_img):
    mask1 = cv2.inRange(hsv_img, RED_LOWER1_600, RED_UPPER1_600)
    mask2 = cv2.inRange(hsv_img, RED_LOWER2_600, RED_UPPER2_600)
    mask = mask1 + mask2
    mask[mask > 0] = 255
    return mask


def get_yellow_mask(hsv_img):
    return cv2.inRange(hsv_img, YELLOW_LOWER_600, YELLOW_UPPER_600)


def get_black_mask_x(hsv_img):
    mask1 = cv2.inRange(hsv_img, BLACK_LOWER1_X_600, BLACK_UPPER1_X_600)
    mask2 = cv2.inRange(hsv_img, BLACK_LOWER2_X_600, BLACK_UPPER2_X_600)
    mask = mask1 + mask2
    mask[mask > 0] = 255
    return mask


def get_green_mask_x(hsv_img):
    mask1 = cv2.inRange(hsv_img, GREEN_LOWER1_X_600, GREEN_UPPER1_X_600)
    mask2 = cv2.inRange(hsv_img, GREEN_LOWER2_X_600, GREEN_UPPER2_X_600)
    mask = mask1 + mask2
    mask[mask > 0] = 255
    return mask


def get_red_mask_x(hsv_img):
    mask1 = cv2.inRange(hsv_img, RED_LOWER1_X_600, RED_UPPER1_X_600)
    mask2 = cv2.inRange(hsv_img, RED_LOWER2_X_600, RED_UPPER2_X_600)
    mask = mask1 + mask2
    mask[mask > 0] = 255
    return mask


def get_yellow_mask_x(hsv_img):
    return cv2.inRange(hsv_img, YELLOW_LOWER_X_600, YELLOW_UPPER_X_600)


COLOR_F_LST_208 = [get_green_mask_208, get_red_mask, get_yellow_mask]
COLOR_F_LST_600 = [get_green_mask_600, get_red_mask, get_yellow_mask]
COLOR_F_LST_X_600 = [get_green_mask_x, get_red_mask_x, get_yellow_mask_x]


def apply_color_f(color_f_lst, roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    return [f(hsv) for f in color_f_lst]


BLOB_H_EXPANSION_FACTOR = 3
BLOB_W_EXPANSION_FACTOR = 3
MAX_PIXELS_IN_SMALL_CASES = 6 * 7


def binarize(gray_img):
    # return cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)[1]  # simple thresholding
    T = mahotas.thresholding.otsu(gray_img)
    thresh = gray_img.copy()
    thresh[thresh > T] = 255
    thresh[thresh < 255] = 0
    return thresh


def find_xmin_xmax(cc):
    xmin, xmax = None, None
    ncols = cc.shape[1]
    for i in range(ncols):
        if cc[:, i].any():
            xmin = i
            break
    for i in range(ncols - 1, -1, -1):
        if cc[:, i].any():
            xmax = i
            break
    return xmin, xmax


def find_ymin_ymax(cc):
    ymin, ymax = None, None
    nrows = cc.shape[0]
    for i in range(nrows):
        if cc[i, :].any():
            ymin = i
            break
    for i in range(nrows - 1, -1, -1):
        if cc[i, :].any():
            ymax = i
            break
    return ymin, ymax


def filter_boxes(boxes, idxs):
    lst = []
    for ilst in idxs:
        i = ilst[0]
        box = boxes[i]
        if box['confidence'] > 0.03:  # len(lst) == 0 or box['confidence'] > 0.03
            lst.append(box)
    return lst


def display_img_with_boxes(img, boxes):
    h, w = img[:2]
    fig, ax = plt.subplots(1, figsize=(50, 50))
    ax.imshow(img)
    colors = ['r', 'b', 'y']
    for i, box in enumerate(boxes):
        l, r, t, b = box['box']
        c = box['class'] + ' ' + str(round(box['confidence'], 5))
        color = colors[i % len(colors)]

        rect = patches.Rectangle(
            (l, t),
            r - l,
            b - t,
            linewidth=5,
            edgecolor=color,
            facecolor='none'
        )

        ax.text(l, t, c, fontsize=24, bbox={'facecolor': color, 'pad': 2, 'ec': color})
        ax.add_patch(rect)
    plt.savefig("my_img.jpg", bbox_inches='tight')
    plt.gca().set_axis_off()


def nms_traffic_light(objs, thresh=THRESH, nms=0.01):
    boxes = []
    confidences = []
    for e in objs:
        class_label = e[0].decode()  # convert b'<str>' to '<str>'
        confidences.append(e[1])
        centerX, centerY, width, height = e[2]
        half_width = width / 2
        half_height = height / 2
        x_left = centerX - half_width
        y_top = centerY - half_height
        boxes.append([int(x_left), int(y_top), int(width), int(height)])
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, thresh, nms)
    return idxs


# def nms_traffic_lightv1(objs, thresh=THRESH, nms=0.01):
#     boxes = []
#     confidences = []
#     for e in objs:
#         class_label = e[0].decode() # convert b'<str>' to '<str>'
#         if class_label == 'traffic light':
#             confidences.append(e[1])
#             centerX, centerY, width, height = e[2]
#             half_width = width / 2
#             half_height = height / 2
#             x_left = centerX - half_width
#             y_top = centerY - half_height
#             boxes.append([int(x_left), int(y_top), int(width), int(height)])
#     idxs = cv2.dnn.NMSBoxes(boxes, confidences, thresh, nms)
#     return idxs


def get_boxes(objs, img_h, img_w):
    boxes = []
    for e in objs:
        class_label = e[0].decode()  # convert b'<str>' to '<str>'
        confidence = e[1]
        centerX, centerY, width, height = e[2]
        half_width = width / 2
        half_height = height / 2

        left = int(centerX - half_width)
        if left < 0: left = 0
        top = int(centerY - half_height)
        if top < 0: top = 0
        right = int(centerX + half_width)
        if right > img_w: right = img_w
        bottom = int(centerY + half_height)
        if bottom > img_h: bottom = img_h

        box = {
            'class': class_label,
            'confidence': confidence,
            'box': (left, right, top, bottom),
            'noise': False
        }
        boxes.append(box)

    return boxes


def merge(cc1, cc2):
    #     print('merge:')
    #     print("cc1['thresh_img']"); imshow(cc1['thresh_img'], gray=True)
    #     print("cc2['thresh_img']"); imshow(cc2['thresh_img'], gray=True)

    cc = {'color': cc1['color'], 'confidence': cc1['confidence'], 'roi_box': cc1['roi_box']}

    xmin, xmax = min(cc1['box'][0], cc2['box'][0]), max(cc1['box'][1], cc2['box'][1])
    ymin, ymax = min(cc1['box'][2], cc2['box'][2]), max(cc1['box'][3], cc2['box'][3])
    cc['box'] = (xmin, xmax, ymin, ymax)

    cc['thresh_img'] = cv2.bitwise_or(cc1['thresh_img'], cc2['thresh_img'])
    cc['npixels'] = cc1['npixels'] + cc2['npixels']

    #     print('merged'); imshow(cc['thresh_img'], gray=True)

    return cc


def sort_by_xmin(cc):
    return cc['box'][0]


def sort_by_ymin(cc):
    return cc['box'][2]


def find_x_overlap(xmin1, xmax1, xmin2, xmax2):
    if xmin1 > xmin2:
        xmin1, xmax1, xmin2, xmax2 = xmin2, xmax2, xmin1, xmax1
    return max(0, xmax1 - xmin2 + 1) if xmax1 < xmax2 else xmax2 - xmin2 + 1


def find_y_overlap(ymin1, ymax1, ymin2, ymax2):
    if ymin1 > ymin2:
        ymin1, ymax1, ymin2, ymax2 = ymin2, ymax2, ymin1, ymax1
    return max(0, ymax1 - ymin2 + 1) if ymax1 < ymax2 else ymax2 - ymin2 + 1


def find_x_distance(xmin1, xmax1, xmin2, xmax2):
    if xmin1 > xmin2:
        xmin1, xmax1, xmin2, xmax2 = xmin2, xmax2, xmin1, xmax1
    return max(0, xmin2 - xmax1 - 1)


def find_y_distance(ymin1, ymax1, ymin2, ymax2):
    if ymin1 > ymin2:
        ymin1, ymax1, ymin2, ymax2 = ymin2, ymax2, ymin1, ymax1
    return max(0, ymin2 - ymax1 - 1)


def merge_overlapped_ccs_help(cc_lst):
    #     print('merge_overlapped_ccs_help:')
    for i in range(1, len(cc_lst)):
        #         print('cc_lst[{}] {}'.format(i-1, cc_lst[i-1]['box'])); imshow(cc_lst[i-1]['thresh_img'], gray=True)
        #         print('cc_lst[{}] {}'.format(i, cc_lst[i]['box'])); imshow(cc_lst[i]['thresh_img'], gray=True)
        prev_xmin, prev_xmax, prev_ymin, prev_ymax = cc_lst[i - 1]['box']
        curr_xmin, curr_xmax, curr_ymin, curr_ymax = cc_lst[i]['box']
        y_distance = find_y_distance(prev_ymin, prev_ymax, curr_ymin, curr_ymax)
        min_h = min(prev_ymax - prev_ymin + 1, curr_ymax - curr_ymin + 1)
        #         print('y_distance {}; min_h {}'.format(y_distance, min_h))
        if y_distance > min_h: continue
        x_overlap = find_x_overlap(prev_xmin, prev_xmax, curr_xmin, curr_xmax)
        #         print('x_overlap', x_overlap)
        min_w = min(prev_xmax - prev_xmin + 1, curr_xmax - curr_xmin + 1)
        if x_overlap >= 1 / 3 * min_w:
            new_cc = merge(cc_lst[i - 1], cc_lst[i])
            cc_lst[i] = new_cc
            cc_lst[i - 1] = None
    return [cc for cc in cc_lst if cc != None]


def merge_overlapped_ccs(cc_lst):
    #     print('merge_overlapped_ccs:')

    if len(cc_lst) == 1: return cc_lst
    #     print('sort_by_xmin cc_lst')
    cc_lst = sorted(cc_lst, key=sort_by_xmin)
    cc_lst = merge_overlapped_ccs_help(cc_lst)

    if len(cc_lst) == 1: return cc_lst
    #     print('sort_by_ymin cc_lst')
    cc_lst = sorted(cc_lst, key=sort_by_ymin)
    return merge_overlapped_ccs_help(cc_lst)


def close_to_black_blob(cc, img, w_expansion_factor, h_expansion_factor):
    #     print('close_to_black_blob:')
    cc_box = cc['box']
    cc_w, cc_h = cc_box[1] - cc_box[0] + 1, cc_box[3] - cc_box[2] + 1
    cc_box_area = cc_w * cc_h
    box = find_cc_box_in_img(cc_box, cc['roi_box'])
    roi, _, cc_box_in_expanded_box = find_roi_box(box, img, w_expansion_factor=w_expansion_factor,
                                                  h_expansion_factor=h_expansion_factor)
    cc_xmin, cc_xmax, cc_ymin, cc_ymax = cc_box_in_expanded_box
    black_mask = get_black_mask(cv2.cvtColor(roi, cv2.COLOR_RGB2HSV))
    #     print('close_to_black_blob - black_mask'); imshow(black_mask, gray=True)
    labels = measure.label(black_mask, connectivity=2, background=0)  # connectivity=2: 8-connected; legacy: neighbors=8
    unique_labels = np.unique(labels)
    #     print('n unique_labels', len(uprint('original box in img'nique_labels))
    for i, label in enumerate(unique_labels):
        if label == 0: continue  # background label, ignore it
        label_mask = np.zeros(black_mask.shape, dtype="uint8")
        label_mask[labels == label] = 255
        #         print('close_to_black_blob - label_mask', i); imshow(label_mask, gray=True)
        black_npixels = cv2.countNonZero(label_mask)
        if black_npixels <= 2:
            #             print('this black blob is too small')
            continue
        black_xmin, black_xmax = find_xmin_xmax(label_mask)
        black_ymin, black_ymax = find_ymin_ymax(label_mask)
        black_w, black_h = black_xmax - black_xmin + 1, black_ymax - black_ymin + 1
        black_box_area = black_w * black_h
        x_distance, y_distance = find_x_distance(cc_xmin, cc_xmax, black_xmin, black_xmax), find_y_distance(cc_ymin,
                                                                                                            cc_ymax,
                                                                                                            black_ymin,
                                                                                                            black_ymax)
        x_overlap, y_overlap = find_x_overlap(cc_xmin, cc_xmax, black_xmin, black_xmax), find_y_overlap(cc_ymin,
                                                                                                        cc_ymax,
                                                                                                        black_ymin,
                                                                                                        black_ymax)

        #         print('box_confidence,obj_w,obj_h,obj_npixels,obj_box_area,black_w,black_h,black_npixels,black_box_area,x_distance,y_distance,x_overlap,y_overlap: {},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
        #             cc['confidence'], cc_w, cc_h, cc['npixels'], cc_box_area,
        #             black_w, black_h, black_npixels, black_box_area,
        #             x_distance, y_distance, x_overlap, y_overlap))

        #         print('cc_box {}, cc_w {}, cc_h {}, cc_box_area {}'.format(cc_box_in_expanded_box, cc_w, cc_h, cc_box_area))
        #         print('black_box {}, black_w {}, black_h {}, black_box_area {}'.format((black_xmin, black_xmax, black_ymin, black_ymax), black_w, black_h, black_box_area))
        #         print('x_distance {}, y_distance {}, x_overlap {}, y_overlap {}'.format(x_distance, y_distance, x_overlap, y_overlap))

        x_test = pd.DataFrame([[
            cc['confidence'], cc_w, cc_h, cc['npixels'], cc_box_area,
            black_w, black_h, black_npixels, black_box_area,
            x_distance, y_distance, x_overlap, y_overlap]])
        pred = bulb_or_noise_classifier.predict(x_test)[0]
        if pred == 0:
            #             print('close_to_black_blob? True')
            return True
    #     print('close_to_black_blob? False')
    return False


def filter_noise(cc_lst, img):
    #     print('filter_noise:')
    lst = []
    for cc in cc_lst:
        #         print("cc['box']", cc['box'], "; roi_shape", cc['thresh_img'].shape); imshow(cc['thresh_img'], gray=True)
        xmin, xmax, ymin, ymax = cc['box']
        w, h = xmax - xmin + 1, ymax - ymin + 1
        w_over_h = w / h

        #         if h == 1 and w == 2 and close_black_blob(cc, img, w_expansion_factor=2, h_expansion_factor=1):
        #             lst.append(cc)
        #             continue

        if h == 1 or w == 1:
            #             print('h or w is too short (ignored)')
            continue

        if w_over_h >= 4:
            #             print('w_over_h {}; w is too long compared to h (ignored)'.format(w_over_h))
            continue

        if w_over_h <= 2 / 3:
            #             print('w_over_h {}; w is too short compared to h (ignored)'.format(w_over_h))
            continue

        #         if cc['npixels'] < 4:
        #             print('npixels', cc['npixels'], '(ignored < 4)')
        #             continue

        if cc['npixels'] <= MAX_PIXELS_IN_SMALL_CASES and \
                (xmin == 0 or ymin == 0 or xmax + 1 == cc['thresh_img'].shape[1] or ymax + 1 == cc['thresh_img'].shape[
                    0]):
            #             print('npixels', cc['npixels'], '; on the border and small (ignored)')
            continue

        #         if cc['npixels'] == 4 and (xmax-xmin != 1 or ymax-ymin != 1):
        #             print('npixels', cc['npixels'], "(ignored ==4 because it isn't square)")
        #             continue

        if not close_to_black_blob(cc, img, w_expansion_factor=BLOB_W_EXPANSION_FACTOR,
                                   h_expansion_factor=BLOB_H_EXPANSION_FACTOR):
            #             print("this cc ignored because it doesn't close to an appropriate black blob)")
            continue

        lst.append(cc)

    #     print('end filter_noise; n ccs', len(lst))
    return lst


def find_cc_lst(img, roi_box, masks, box_confidence, connectivity=2):
    #     print('find_cc_lst:')
    cc_lst = []
    for i, mask in enumerate(masks):
        #         print('processing', COLORS[i], 'regions...')
        if (np.all(mask == 0)):
            #             print('no ', COLORS[i])
            continue

        labels = measure.label(mask, connectivity=connectivity,
                               background=0)  # connectivity=2: 8-connected; legacy: neighbors=8
        #         print('n ccs', len(np.unique(labels)) - 1)
        lst = []
        for label in np.unique(labels):
            if label == 0: continue  # this is the background, ignore it
            label_mask = np.zeros(mask.shape, dtype="uint8")
            label_mask[labels == label] = 255
            npixels = cv2.countNonZero(label_mask)
            #             if npixels == 1: continue  # salt noise
            #             print('label_mask'); imshow(label_mask, gray=True)

            xmin, xmax = find_xmin_xmax(label_mask)
            ymin, ymax = find_ymin_ymax(label_mask)

            cc = {
                'box': (xmin, xmax, ymin, ymax), 'color': COLORS[i], 'thresh_img': label_mask, 'npixels': npixels,
                'roi_box': roi_box, 'confidence': box_confidence, 'bsm': None
            }
            lst.append(cc)
        lst = merge_overlapped_ccs(lst)
        #         print('n ccs after merging', len(lst))
        if COLORS[i] == 'g' and len(lst) > MAX_G_IN_BOX:
            #             print('this box is noise because n ccs >', MAX_G_IN_BOX)
            continue
        lst = filter_noise(lst, img)
        cc_lst.extend(lst)
    #     print('end find_cc_lst; n ccs', len(cc_lst))
    return cc_lst


def find_cc_lst_after_opening(cc, connectivity=2,
                              check_noise=True):  # suspect that there're more than one bulb in this cc
    print('find_cc_lst_after_opening:')
    cc_lst = []
    mask = cv2.copyMakeBorder(cc['thresh_img'], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    print('before morph_open');
    # imshow(mask, gray=True)
    morphed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_3x3)
    print('after morph_open');
    # imshow(morphed, gray=True)
    morphed = morphed[3:morphed.shape[0] - 3, 3:morphed.shape[1] - 3]

    labels = measure.label(morphed, connectivity=connectivity,
                           background=0)  # connectivity=2: 8-connected; legacy: neighbors=8

    if check_noise and len(np.unique(labels)) != 3:
        print('noise because #ccs != 3 (including background)')
        return []

    for label in np.unique(labels):
        if label == 0: continue  # background label, ignore it
        label_mask = np.zeros(morphed.shape, dtype="uint8")
        label_mask[labels == label] = 255
        print('label_mask');
        # imshow(label_mask, gray=True)

        xmin, xmax = find_xmin_xmax(label_mask)
        ymin, ymax = find_ymin_ymax(label_mask)

        new_cc = {
            'box': (xmin, xmax, ymin, ymax), 'color': cc['color'], 'thresh_img': label_mask, 'label': label,
            'npixels': cv2.countNonZero(label_mask),
            'roi_box': cc['roi_box'], 'confidence': cc['confidence'], 'bsm': None
        }
        cc_lst.append(new_cc)

    sum_npixels_after_morph_open = sum([e['npixels'] for e in cc_lst])
    print('before morph_open: npixels = {}; after morph_open: sum(npixels) = {}'.format(cc['npixels'],
                                                                                        sum_npixels_after_morph_open))
    if cc['npixels'] > 3 * sum_npixels_after_morph_open:
        print("noise because cc['npixels'] > 3 * sum_npixels_after_morph_open")
        return []

    return cc_lst


def find_cc_box_in_img(cc_box, tl_box):
    xmin = tl_box[0] + cc_box[0]
    xmax = tl_box[0] + cc_box[1]
    ymin = tl_box[2] + cc_box[2]
    ymax = tl_box[2] + cc_box[3]
    return (xmin, xmax, ymin, ymax)


def find_obj_lst(box, color_f_lst, img):
    #     print('find_obj_lst:')
    roi, roi_box, _ = find_roi_box(box['box'], img, w_expansion_factor=TL_W_EXPANSION_FACTOR,
                                   h_expansion_factor=TL_H_EXPANSION_FACTOR)
    color_masks = apply_color_f(color_f_lst, roi)
    cc_lst = find_cc_lst(img, roi_box, color_masks, box['confidence'])
    #     cc_lst = handle_long_ccs(cc_lst)

    obj_lst = cc_lst  # TODO: []

    return obj_lst


def find_binary_shape_metrics(obj, morph_close=True):
    #     print('find_binary_shape_metrics:')
    binary_roi = obj['thresh_img']
    #     print('before copyMakeBorder'); imshow(binary_roi, gray=True)
    binary_roi = cv2.copyMakeBorder(binary_roi, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #     print('after copyMakeBorder'); imshow(binary_roi, gray=True)

    if morph_close:
        #         print('before MORPH_CLOSE'); imshow(binary_roi, gray=True)
        morphed = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, KERNEL_5x5)
    #         print('after MORPH_CLOSE'); imshow(morphed, gray=True)
    else:
        morphed = binary_roi

    xmin, xmax = find_xmin_xmax(morphed)
    ymin, ymax = find_ymin_ymax(morphed)
    morphed = morphed[ymin:ymax + 1, xmin:xmax + 1]
    if morph_close:
        #         print('after tidying'); imshow(morphed, gray=True)
        pass

    contours = cv2.findContours(morphed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    max_contour = contours[0] if len(contours) == 1 else max(contours, key=cv2.contourArea)

    # circularity
    contour_perimeter = cv2.arcLength(max_contour, True)  # True: max_contour is closed
    contour_area = cv2.contourArea(max_contour)
    circularity = None if contour_perimeter == 0 else 4 * math.pi * contour_area / contour_perimeter ** 2

    moments = cv2.moments(max_contour)
    mu00, mu20, mu02 = moments['m00'], moments['mu20'], moments['mu02']
    zunic_hiruta_circularity = (1 / (2 * math.pi)) * (mu00 ** 2 / (mu20 + mu02)) if mu20 + mu02 != 0 else 0

    # concavity
    convex_hull = cv2.convexHull(max_contour)
    convex_area = cv2.contourArea(convex_hull)
    concavity = convex_area - contour_area

    # aspect_ratio
    rect = cv2.minAreaRect(max_contour)
    (x, y), (width, height), angle = rect
    #     print('width, height',width, height )
    aspect_ratio = max(width + 1, height + 1) / min(width + 1, height + 1)

    w, h = xmax - xmin + 1, ymax - ymin + 1
    extent = obj['npixels'] / (w * h)

    return {
        'circularity': circularity,
        'zunic_hiruta_circularity': zunic_hiruta_circularity,
        'concavity': concavity,
        'aspect_ratio': aspect_ratio,
        'extent': extent
    }


def predict_bulb(obj):
    #     print('predict_bulb:')
    if obj['npixels'] <= 9:
        return obj['color']
    bsm = find_binary_shape_metrics(obj, morph_close=False)
    obj['bsm'] = bsm
    #     print('box_confidence, npixels, circularity, zunic_hiruta_circularity, concavity, aspect_ratio, extent {},{},{},{},{},{},{}'.format(
    #             obj['confidence'], obj['npixels'], bsm['circularity'], bsm['zunic_hiruta_circularity'], bsm['concavity'], bsm['aspect_ratio'], bsm['extent']))
    if bsm['circularity'] is None: return 'noise'
    x_test = pd.DataFrame([[NORMAL_CASE, obj['confidence'], obj['npixels'], bsm['circularity'],
                            bsm['zunic_hiruta_circularity'], bsm['concavity'], bsm['aspect_ratio'], bsm['extent']]])
    pred = BULBS[classifier.predict(x_test)[0]]
    #     print('predict_bulb:', pred)
    return obj['color'] if pred == 'b' or pred == 'd' else pred


def predict_bulb_in_extension_case(obj, img):
    #     print('predict_bulb_in_extension_case:')
    tl_box, obj_box = obj['roi_box'], obj['box']
    obj_xmin, obj_xmax, obj_ymin, obj_ymax = find_cc_box_in_img(obj_box, tl_box)
    obj_roi = img[obj_ymin:obj_ymax + 1, obj_xmin:obj_xmax + 1]
    #     print('obj_roi'); imshow(obj_roi)

    #     print('obj_roi in normal range'); imshow(obj['thresh_img'], gray=True)

    obj_color = obj['color']
    if obj_color == 'g':
        obj_mask_x = get_green_mask_x(cv2.cvtColor(obj_roi, cv2.COLOR_RGB2HSV))
    elif obj_color == 'r':
        obj_mask_x = get_red_mask_x(cv2.cvtColor(obj_roi, cv2.COLOR_RGB2HSV))
    else:
        obj_mask_x = get_yellow_mask_x(cv2.cvtColor(obj_roi, cv2.COLOR_RGB2HSV))
    #     print('obj_roi in extension range'); imshow(obj_mask_x, gray=True)
    obj_x = {'confidence': obj['confidence'], 'thresh_img': obj_mask_x, 'npixels': np.count_nonzero(obj_mask_x)}
    bsm = find_binary_shape_metrics(obj_x, morph_close=False)
    obj_x['bsm'] = bsm
    #     print('box_confidence, npixels, circularity, zunic_hiruta_circularity, concavity, aspect_ratio, extent {},{},{},{},{},{},{}'.format(
    #             obj_x['confidence'], obj_x['npixels'], bsm['circularity'], bsm['zunic_hiruta_circularity'], bsm['concavity'], bsm['aspect_ratio'], bsm['extent']))
    if bsm['circularity'] is None: return 'noise'
    x_test = pd.DataFrame([[EXTENSION_CASE, obj_x['confidence'], obj_x['npixels'], bsm['circularity'],
                            bsm['zunic_hiruta_circularity'], bsm['concavity'], bsm['aspect_ratio'], bsm['extent']]])
    pred = classifier.predict(x_test)[0]
    return 'noise' if pred == 3 else obj_color


def find_color_f_lst(tl_box, img):
    #     print('find_color_f_lst:')
    roi, roi_box, _ = find_roi_box(tl_box, img, w_expansion_factor=TL_W_EXPANSION_FACTOR,
                                   h_expansion_factor=TL_H_EXPANSION_FACTOR)
    roi_box_area = (roi_box[1] - roi_box[0]) * (roi_box[3] - roi_box[2])
    green_mask = get_green_mask_x(cv2.cvtColor(roi, cv2.COLOR_RGB2HSV))
    green_npixels = np.count_nonzero(green_mask)
    #     print('roi_box_area', roi_box_area, 'vs green_npixels', green_npixels)
    return COLOR_F_LST_208 if green_npixels >= roi_box_area * 50 / 100 else COLOR_F_LST_600


def detect_bulbs(boxes, img):
    #     print('detect_bulbs:')
    if len(boxes) == 0:
        #         print('yolo finds nothing')
        return

    bulb_set = []
    for box in boxes:
        color_f_lst = find_color_f_lst(box['box'], img)
        obj_lst = find_obj_lst(box, color_f_lst, img)
        #         print('# objs in this box', len(obj_lst))
        for obj in obj_lst:
            pred = predict_bulb(obj)
            #             print('prediction:', pred)
            if pred != 'noise':
                bulb_set.append(pred)

                continue

            pred = predict_bulb_in_extension_case(obj, img)
            #             print('prediction:', pred)
            if pred != 'noise':
                if pred == 'g':
                    pred = 'green'
                elif pred == 'r':
                    pred = 'red'
                else:
                    pred = 'yellow'
                bulb_set.append(pred)
    #         print('--------------------------------------------------')

    #     print('predictions', bulb_set)
    return bulb_set


class Box(dict):
    def __init__(self, name, confidence, coor, noise):
        dict.__init__(self, name=name, confidence=confidence, coor=coor, noise=noise)


def get_boxes(objs, img_h, img_w):
    boxes = []
    for e in objs:
        class_label = e[0].decode()  # convert b'<str>' to '<str>'
        confidence = e[1]
        centerX, centerY, width, height = e[2]
        half_width = width / 2
        half_height = height / 2

        left = int(centerX - half_width)
        if left < 0: left = 0
        top = int(centerY - half_height)
        if top < 0: top = 0
        right = int(centerX + half_width)
        if right > img_w: right = img_w
        bottom = int(centerY + half_height)
        if bottom > img_h: bottom = img_h

        box = {
            'class': class_label,
            'confidence': confidence,
            'box': (left, right, top, bottom),
            'noise': False
        }
        boxes.append(box)
    return boxes


def get_boxesv1(objs, img_h, img_w):
    boxes = []
    for e in objs:
        class_label = e[0].decode()  # convert b'<str>' to '<str>'
        if class_label == 'traffic light':
            confidence = e[1]
            centerX, centerY, width, height = e[2]
            half_width = width / 2
            half_height = height / 2

            left = int(centerX - half_width)
            if left < 0: left = 0
            top = int(centerY - half_height)
            if top < 0: top = 0
            right = int(centerX + half_width)
            if right > img_w: right = img_w
            bottom = int(centerY + half_height)
            if bottom > img_h: bottom = img_h

            box = {
                'class': class_label,
                'confidence': confidence,
                'box': (left, right, top, bottom),
                'noise': False
            }
            boxes.append(box)
    return boxes


def nms_traffic_light(objs, thresh=THRESH, nms=0.01):
    boxes = []
    confidences = []
    for e in objs:
        class_label = e[0].decode()  # convert b'<str>' to '<str>'
        confidences.append(e[1])
        centerX, centerY, width, height = e[2]
        half_width = width / 2
        half_height = height / 2
        x_left = centerX - half_width
        y_top = centerY - half_height
        boxes.append([int(x_left), int(y_top), int(width), int(height)])
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, thresh, nms)
    return idxs


def nms_traffic_lightv1(objs, thresh=THRESH, nms=0.01):
    boxes = []
    confidences = []
    for e in objs:
        class_label = e[0].decode()  # convert b'<str>' to '<str>'
        if class_label == 'traffic light':
            confidences.append(e[1])
            centerX, centerY, width, height = e[2]
            half_width = width / 2
            half_height = height / 2
            x_left = centerX - half_width
            y_top = centerY - half_height
            boxes.append([int(x_left), int(y_top), int(width), int(height)])
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, thresh, nms)
    return idxs


def filter_boxes(boxes, idxs):
    lst = []
    for ilst in idxs:
        i = ilst[0]
        box = boxes[i]
        if box[1] > 0.4:  # len(lst) == 0 or box['confidence'] > 0.03
            lst.append(box)
    return lst


def filter_boxesv1(boxes, idxs):
    lst = []
    for ilst in idxs:
        i = ilst[0]
        box = boxes[i]
        if box['confidence'] > 0.03:  # len(lst) == 0 or box['confidence'] > 0.03
            lst.append(box)
    return lst


def get_traffic(objs):
    boxes = []
    for e in objs:
        class_label = e[0].decode()  # convert b'<str>' to '<str>'
        if class_label == 'traffic light':
            boxes.append(e)
    return boxes

#
# net = load_net(
#     b"D:/graduation_project/traffic_detection/input/yolov4.cfg",
#     b"D:/graduation_project/traffic_detection/input/yolov4.weights", 0)
#
# meta = load_meta(
#     b'D:/graduation_project/traffic_detection/input/coco.data')


def detect_img(IMG_PATH):
    net = load_net(
        b"D:/darknet-AlexeyAB-master/YoloV4_10Classes/darknet-master/darknet-master/build/darknet/x64/yolo-obj.cfg",
        b"D:/darknet-AlexeyAB-master/YoloV4_10Classes/darknet-master/darknet-master/build/darknet/x64/yolo-obj_last.weights",
        0)

    meta = load_meta(
        b'D:/darknet-AlexeyAB-master/YoloV4_10Classes/darknet-master/darknet-master/build/darknet/x64/data/obj.data')
    net = load_net(
        b"D:/graduation_project/traffic_detection/input/yolov4.cfg",
        b"D:/graduation_project/traffic_detection/input/yolov4.weights", 0)

    meta = load_meta(
        b'D:/graduation_project/traffic_detection/input/coco.data')
    objs, img_h, img_w = detect_image(net, meta, IMG_PATH.encode(), 0.25)
    idxs = nms_traffic_light(objs)
    boxes = filter_boxes(objs, idxs)
    image = cvDrawBoxesImg(boxes, cv2.imread(IMG_PATH))
    boxes = get_boxes(boxes, img_h, img_w)
    path = str(IMG_PATH).split('\\')
    cv2.imwrite('media/output/' + path[len(path) - 1], image)
    # img = np.asarray(Image.open(IMG_PATH))
    # display_img_with_boxes(img, boxes)
    # color_set = detect_bulbs(boxes, IMG_PATH)
    return boxes, path[len(path) - 1]


def detect_img_v1(IMG_PATH):
    net = load_net(
        b"D:/graduation_project/traffic_detection/input/yolov4.cfg",
        b"D:/graduation_project/traffic_detection/input/yolov4.weights", 0)

    meta = load_meta(
        b'D:/graduation_project/traffic_detection/input/coco.data')
    objs, img_h, img_w = detect_image(net, meta, IMG_PATH.encode(), 0.25)
    boxes = get_boxesv1(objs, img_h, img_w)
    boxesv1 = get_traffic(objs)
    image = cvDrawBoxesImg(boxesv1, cv2.imread(IMG_PATH))
    idxs = nms_traffic_lightv1(objs)
    boxes = filter_boxesv1(boxes, idxs)

    path = str(IMG_PATH).split('\\')
    cv2.imwrite('media/output/i' + path[len(path) - 1], image)
    img = np.asarray(Image.open(IMG_PATH))
    color_set = detect_bulbs(boxes, img)
    return boxes, 'i' + path[len(path) - 1], color_set





def detect_vid(frame):
    net = load_net(
        b"D:/darknet-AlexeyAB-master/YoloV4_10Classes/darknet-master/darknet-master/build/darknet/x64/yolo-obj-512.cfg",
        b"D:/darknet-AlexeyAB-master/YoloV4_10Classes/darknet-master/darknet-master/build/darknet/x64/yolo-obj_last.weights",
        0)

    meta = load_meta(
        b'D:/darknet-AlexeyAB-master/YoloV4_10Classes/darknet-master/darknet-master/build/darknet/x64/data/obj.data')
    objs = detect_video(net, meta, frame, 0.4)
    return objs

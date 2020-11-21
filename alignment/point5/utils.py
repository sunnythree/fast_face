import sys
import time
import os

def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    if l1 > l2:
        left = l1
    else:
        left = l2
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    if r1 < r2:
        right = r1
    else:
        right = r2
    return right - left

def box_intersection(a, b):
    w = overlap(a[0], a[2], b[0], b[2])
    h = overlap(a[1], a[3], b[1], b[3])
    if w < 0 or h < 0:
        return 0
    area = w * h
    return area


def box_union(a, b):
    i = box_intersection(a, b)
    u = a[2] * a[3] + b[2] * b[3] - i
    return u


def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def confidence_box(a):
    return a[0]

def nms(bboxes, confidence, iou_thresh):
    nms_boxes = []
    bboxes = sorted(bboxes, key=confidence_box, reverse=True)
    for box in bboxes:
        if box[0] < confidence:
            continue
        is_contain = False
        for i in range(len(nms_boxes)):
            nms_box = nms_boxes[i]
            iou = box_iou(box, nms_box)
            if iou > iou_thresh:
                is_contain = True
        if not is_contain:
            nms_boxes.append(box)
    return nms_boxes


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

import xml.etree.ElementTree as ET
import os
import sys

classes = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
           "16","17","18","19","20","21","22","23","24","25","26","27","28","29","30",
           "31","32","33","34","35","36","37","38","39","40","41","42","43","44","45",
           "46","47","48","49","50","51","52","53","54","55","56","57","58","59"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(input_file, output_file):
    in_file = open(input_file)
    out_file = open(output_file, 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

if __name__=='__main__':
    input_dir = 'C:/Users/javer/Downloads/yolo/outputs'
    output_dir = 'C:/Users/javer/Downloads/yolo/yolo'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_list = os.listdir(input_dir)
    for file in file_list:
        convert_annotation(os.path.join(input_dir, file), os.path.join(output_dir,file.replace('xml', 'txt')))
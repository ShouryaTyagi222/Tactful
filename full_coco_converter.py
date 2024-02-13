from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import random
import cv2
import os
import json
import argparse

output_path='/data/circulars/DATA/TACTFUL/Data'
data_file='/data/circulars/DATA/Models/CircularsV1/docvqa.json'

with open(data_file, 'r') as f:
    data = json.load(f)

coco = Coco()
map = {'Date Block': 0, 'Logos': 1, 'Subject Block': 2, 'Body Block': 3, 'Circular ID': 4, 'Table': 5, 'Stamps-Seals': 6, 'Handwritten Text': 7, 'Copy-Forwarded To Block': 8, 'Address of Issuing Authority': 9, 'Signature': 10, 'Reference Block': 11, 'Signature Block': 12, 'Header Block': 13, 'Addressed To Block': 14}
for k,v in map.items():
  coco.add_category(CocoCategory(id=v, name=k, supercategory='first'))
# coco.add_category(CocoCategory(id=1, name='vehicle'))

labels=set()
q_labels=set()
files=[]


# d2 = {'Signature': 'Signature', 'Copy-Forwarded To Block': 'Copy-Forwarded To Block', 'Stamps-Seals': 'Stamps-Seals', 'Reference Block': 'Reference Block', 'Circular ID': 'Circular ID', 'Date Block': 'Date Block', 'Header Block': 'Header Block', 'Handwritten Text': 'Handwritten Text', 'Addressed To Block': 'Addressed To Block', 'Logos': 'Logos', 'Signature Block': 'Signature Block', 'Address of Issuing Authority': 'Address of Issuing Authority', 'Subject Block': 'Subject Block', 'Table': 'Table', 'Body Block': 'Body Block'}


for da in data:
    img_name=da['file_name']
    annotations=da['annotations']
    width=0
    height=0
    files.append(img_name)
    for annotation in annotations:
        if annotation['type']=='rectanglelabels':
            if width==0:
                width=annotation['original_width']
                height=annotation['original_height']
                coco_image = CocoImage(file_name=img_name, height=int(height), width=int(width))
                # print('>>>',[img_name,width,height])
            x,y,w,h=annotation['value']['x'],annotation['value']['y'],annotation['value']['width'],annotation['value']['height']
            label=annotation['value']['rectanglelabels'][0]
            if label in ['Adressed To','Circular Reference','Reference Id']:
                continue
            labels.add(label)
            # print([label,[x,y,w,h]])
            coco_image.add_annotation(CocoAnnotation(bbox=[int(x),int(y),int(w),int(h)], category_id=map[label], category_name=label))
        elif annotation['type']=='textarea':
            value=annotation['value']['text'][0]
            q_label=annotation['to_name']
            q_labels.add(q_label)
    coco.add_image(coco_image)

save_json(data=coco.json, save_path=os.path.join(output_path,'docvqa_coco.json'))
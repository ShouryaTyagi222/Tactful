# pip install sahi
# python coco_converter.py -i '/data/circulars/DATA/Models/CircularsV1/docvqa.json' -d '/data/circulars/DATA/split_circulars/SplitCircularsv2/first_page' -o '/data/circulars/DATA/TACTFUL/Data/docvqa'
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import random
import cv2
import os
import json
import argparse

def main(args):

    # Open the JSON file
    data_file=args.data_file
    img_dir=args.img_dir
    output_dir=args.output_dir
    train_split=float(args.train_split)


    with open(data_file, 'r') as f:
        data = json.load(f)

    train_coco = Coco()
    lake_coco = Coco()
    val_coco = Coco()
    map = {'Date Block': 0, 'Logos': 1, 'Subject Block': 2, 'Body Block': 3, 'Circular ID': 4, 'Table': 5, 'Stamps-Seals': 6, 'Handwritten Text': 7, 'Copy-Forwarded To Block': 8, 'Address of Issuing Authority': 9, 'Signature': 10, 'Reference Block': 11, 'Signature Block': 12, 'Header Block': 13, 'Addressed To Block': 14}
    for k,v in map.items():
        train_coco.add_category(CocoCategory(id=v, name=k))
        lake_coco.add_category(CocoCategory(id=v, name=k))
        val_coco.add_category(CocoCategory(id=v, name=k))

    labels=set()
    q_labels=set()
    files=[]
    tr=0
    la=0
    va=0

    for folder in ['train','lake','val']:
        if not os.path.exists(os.path.join(output_dir,folder)):
            os.mkdir(os.path.join(output_dir,folder))

    for da in data:
        img_name=da['file_name']
        img_path=os.path.join(img_dir,img_name)
        img=cv2.imread(img_path)
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
                x,y,w,h=annotation['value']['x']/100*width,annotation['value']['y']/100*height,annotation['value']['width']/100*width,annotation['value']['height']/100*height
                label=annotation['value']['rectanglelabels'][0]
                if label in ['Adressed To','Circular Reference','Reference Id']:
                    continue
                labels.add(label)
                # print([label,[x,y,w,h]])
                to_be_added_anot=CocoAnnotation(bbox=[int(x),int(y),int(w),int(h)], category_id=map[label], category_name=label)
                if len(to_be_added_anot.bbox)==4:
                    # print(to_be_added_anot.bbox)
                    coco_image.add_annotation(to_be_added_anot)
            elif annotation['type']=='textarea':
                value=annotation['value']['text'][0]
                q_label=annotation['to_name']
                q_labels.add(q_label)

        split = random.randint(0, 2)
       
        if va<=79 and split==2:
            val_coco.add_image(coco_image)
            cv2.imwrite(os.path.join(output_dir,'val',img_name),img)
            va+=1
        elif tr<=45 and split==0:
            train_coco.add_image(coco_image)
            cv2.imwrite(os.path.join(output_dir,'train',img_name),img)
            tr+=1
        else:
            lake_coco.add_image(coco_image)
            cv2.imwrite(os.path.join(output_dir,'lake',img_name),img)
            la+=1

        print(img_name)
        print([tr,va,la])
        print([len(os.listdir(output_dir+'/train')),len(os.listdir(output_dir+'/val')),len(os.listdir(output_dir+'/lake'))])

    save_json(data=train_coco.json, save_path=output_dir+'/docvqa_train_coco.json')
    save_json(data=lake_coco.json, save_path=output_dir+'/docvqa_lake_coco.json')
    save_json(data=val_coco.json, save_path=output_dir+'/docvqa_val_coco.json')

def parse_args():
    parser = argparse.ArgumentParser(description="COCO", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--data_file", default=None, type=str, help="Path to the input json file")
    parser.add_argument("-d", "--img_dir", default=None, type=str, help="Path to the image Directory")
    parser.add_argument("-o", "--output_dir", default='/', type=str, help="Path to the Output Folder")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    main(arg)
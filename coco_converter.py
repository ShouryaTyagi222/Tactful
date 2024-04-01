# pip install sahi
# python coco_converter.py -i '/data/circulars/DATA/LayoutLMV1/docvqa_dataset/raw_data' -d /data/circulars/DATA/LayoutLMV1/docvqa_dataset/Images -o /data/circulars/DATA/TACTFUL/Data/new_faster_rcnn
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import random
import cv2
import os
import json
import argparse
from tqdm import tqdm
import shutil

IMG_WIDTH = 224
IMG_HEIGHT = 224

def main(args):

    # Open the JSON file
    data_file=args.data_file
    img_dir=args.img_dir
    output_dir=args.output_dir

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_coco = Coco()
    lake_coco = Coco()
    val_coco = Coco()
    map = {'Address of Issuing Authority': 0, 'Date Block': 1, 'Header Block': 2, 'Table': 3, 'Circular ID': 4, 'Body Block': 5, 'Signature': 6, 'Signature Block': 7, 'Stamps-Seals': 8, 'Handwritten Text': 9, 'Copy-Forwarded To Block': 10, 'Addressed To Block': 11, 'Subject Block': 12, 'Logos': 13, 'Reference Block': 14, 'Adressed To': 15, 'Circular Reference': 16, 'Name of the signatory': 17, 'Signatory-Designation': 18, 'Reference Id': 19, 'Forwarder': 20, 'Forwarder-Designation': 21, 'Issuing Authority': 22}
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

    data_files = os.listdir(args.data_file)

    final_data = []
    for data_file in data_files:
        with open(os.path.join(args.data_file,data_file), 'r') as f:
            data = json.load(f)
        tr=0
        la=0
        va=0
        for da in tqdm(data):
            img_name=da["data"]['ocr'].split('/')[-1]
            annotations=da['annotations']
            width=0
            height=0
            files.append(img_name)
            for annotation in annotations:
                for annot in annotation['result']:
                    if annot['from_name']=='bbox':
                        continue
                    if annot['type']=='rectanglelabels':
                        if width==0:
                            img = cv2.imread(img_dir + '/' + img_name)
                            img= cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                            y,x = img.shape[:2]
                            width=IMG_WIDTH
                            height=IMG_HEIGHT
                            coco_image = CocoImage(file_name=img_name, height=int(height), width=int(width))
                            # print('>>>',[img_name,width,height])
                        x,y,w,h=annot['value']['x']/100*width,annot['value']['y']/100*height,annot['value']['width']/100*width,annot['value']['height']/100*height
                        label=annot['value']['rectanglelabels'][0]
                        if label=='Stamps/Seals':
                            label='Stamps-Seals'
                        labels.add(label)
                        # print([label,[x,y,w,h]])
                        to_be_added_anot=CocoAnnotation(bbox=[int(x),int(y),int(w),int(h)], category_id=map[label], category_name=label)
                        if len(to_be_added_anot.bbox)==4:
                            # print(to_be_added_anot.bbox)
                            coco_image.add_annotation(to_be_added_anot)

            split = random.randint(0, 2)
        
            if va<50 and split==2:
                val_coco.add_image(coco_image)
                cv2.imwrite(os.path.join(output_dir,'val',img_name),img)
                va+=1
            elif tr<20 and split==0:
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
    parser.add_argument("-d", "--img_dir", default='/', type=str, help="Path to the image Directory")
    parser.add_argument("-o", "--output_dir", default='/', type=str, help="Path to the Output Folder")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    main(arg)
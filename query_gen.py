# python query_gen.py -i '/data/circulars/DATA/split_circulars/SplitCircularsv2/first_page' -c '/data/circulars/DATA/TACTFUL/Data/docvqa_coco.json' -q '/data/circulars/DATA/TACTFUL/Data/query_imgs'
import argparse
import json
import os
import cv2

def main(args):
    with open(args.coco_file, 'r') as f:
        coco = json.load(f)
    img_map={}
    for i in coco['images']:
        img_map[i['id']]=i['file_name']
    MAPPING = {'0': 'Date Block', '1': 'Logos', '2': 'Subject Block', '3': 'Body Block', '4': 'Circular ID', '5': 'Table', '6': 'Stamps-Seals', '7': 'Handwritten Text', '8': 'Copy-Forwarded To Block', '9': 'Address of Issuing Authority', '10': 'Signature', '11': 'Reference Block', '12': 'Signature Block', '13': 'Header Block', '14': 'Addressed To Block'}
    list_of_sets=[set() for i in range(len(MAPPING))]
    for anot in coco['annotations']:
        cls=int(anot['category_id'])
        list_of_sets[cls].add(img_map[anot['image_id']])

    for i in range(len(list_of_sets)):
        list_of_sets[i]=list(list_of_sets[i])
        print(MAPPING[str(i)],':',len(list_of_sets[i]))
    
    query_path = args.query_path
    img_dir=args.img_dir
    for i in range(len(list_of_sets)):
        # if MAPPING[str(i)]!='Stamps/Seals':
        if not os.path.exists(os.path.join(query_path,MAPPING[str(i)])):
            os.mkdir(os.path.join(query_path,MAPPING[str(i)]))
            for img_name in list_of_sets[i]:
                img_path=os.path.join(img_dir,img_name)
                img=cv2.imread(img_path)
                cv2.imwrite(os.path.join(query_path,MAPPING[str(i)],img_name),img)

def parse_args():
    parser = argparse.ArgumentParser(description="Query Image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--img_dir", default=None, type=str, help="Path to the image Directory")
    parser.add_argument("-c", "--coco_file", default=None, type=str, help="Path to the coco file")
    parser.add_argument("-q", "--query_path", default='/', type=str, help="Path to the Output query image Folder")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    main(arg)
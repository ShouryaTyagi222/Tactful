# python query_gen.py -i /data/circulars/DATA/LayoutLMV1/docvqa_dataset/Images -c /data/circulars/DATA/LayoutLMV1/docvqa_dataset/raw_data -q /data/circulars/DATA/TACTFUL/Data/query_images
import argparse
import json
import os
from PIL import Image
from tqdm import tqdm

def main(args):
    data_files = os.listdir(args.data_files_folder)
    if not os.path.exists(args.query_path):
        os.mkdir(args.query_path)
    for data_file in data_files:
        with open(os.path.join(args.data_files_folder,data_file), 'r') as f:
            data = json.load(f)
        
        for da in tqdm(data):
            img_name=da["data"]['ocr'].split('/')[-1]
            img_path=os.path.join(args.img_dir,img_name)
            img=Image.open(img_path)
            annotations=da['annotations']
            for annotation in annotations:
                for annot in annotation['result']:
                    
                    if annot['from_name']=='bbox':
                        continue
                    if annot['type']=='rectanglelabels':
                        label=annot['value']['rectanglelabels'][0]
                        
                        if label=='Stamps/Seals':
                            label='Stamps-Seals'
                        
                        if not os.path.exists(args.query_path+'/'+label):
                            os.mkdir(args.query_path+'/'+label)
                        img.save(os.path.join(args.query_path,label,img_name))

def parse_args():
    parser = argparse.ArgumentParser(description="Query Image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--img_dir", default=None, type=str, help="Path to the image Directory")
    parser.add_argument("-c", "--data_files_folder", default=None, type=str, help="Path to the raw json files folder")
    parser.add_argument("-q", "--query_path", default='/', type=str, help="Path to the Output query image Folder")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    main(arg)
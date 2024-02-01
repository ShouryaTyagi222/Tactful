import os, shutil, json ,cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import numpy as np
from ..configs import *

from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
)
from detectron2.engine import DefaultPredictor, DefaultTrainer

import sys
sys.path.append("../")


class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def create_model(cfg, type="train"):
    if type == "train":
        trainer = CocoTrainer(cfg)
        trainer.resume_or_load(resume=False)
        return trainer
    if type == "test":
        tester = DefaultPredictor(cfg)
        return tester


def crop_object(image, box, ground_truth=False):
    """Crops an object in an image

  Inputs:
    image: PIL image
    box: one box from Detectron2 pred_boxes
    """
    if (not ground_truth):
        x_top_left = box[0]
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]
    else:
        x_top_left = box[0]
        y_top_left = box[1]
        x_bottom_right = box[0] + box[2]
        y_bottom_right = box[1] + box[3]
    x_center = (x_top_left + x_bottom_right) / 2
    y_center = (y_top_left + y_bottom_right) / 2

    try:
        crop_img = image.crop((int(x_top_left), int(y_top_left),
                               int(x_bottom_right), int(y_bottom_right)))
    except Exception as e:
        pass

    return crop_img


def do_evaluate(cfg, model, output_path):
    results = dict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(dataset_name,
                                  output_dir=os.path.join(
                                      output_path, "inference", dataset_name))
        results_i = inference_on_dataset(model.model, data_loader, evaluator)
        results[dataset_name] = results_i
    return results


def remove_dataset(name):
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)


'''
Returns the list of cropped image based on the objects. The method uses the trained object detection\
     model to get bouding box and crop the images.
'''
def crop_images_classwise(model: DefaultPredictor, src_path, dest_path,
                          proposal_budget: int):
    if not os.path.exists(dest_path + '/obj_images'):
        os.makedirs(dest_path + '/obj_images')
    obj_im_dir = dest_path + '/obj_images'
    no_of_objects = 0
    for d in tqdm(os.listdir(src_path)):
        image = cv2.imread(os.path.join(src_path, d))
        height, width = image.shape[:2]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]
        images = model.model.preprocess_image(inputs)

        features = model.model.backbone(images.tensor)
        proposals, _ = model.model.proposal_generator(images, features)
        instances, _ = model.model.roi_heads(images, features,
                                                     proposals)
        boxes = instances[0].pred_boxes
        classes = instances[0].pred_classes.cpu().numpy().tolist()
        max_score_order = torch.argsort(instances[0].scores).tolist()

        if (proposal_budget > len(max_score_order)):
            proposal_budget = len(max_score_order)

        for singleclass in classes:
            if not os.path.exists(
                    os.path.join(dest_path, 'obj_images',
                                 MAPPING[str(singleclass)])):
                os.makedirs(
                    os.path.join(dest_path, 'obj_images',
                                 MAPPING[str(singleclass)]))

        img = Image.open(os.path.join(src_path, d))
        for idx, box in enumerate(
                list(boxes[max_score_order[:proposal_budget]])):
            no_of_objects += 1
            box = box.detach().cpu().numpy()

            crop_img = crop_object(img, box)
            try:
                crop_img.save(
                    os.path.join(
                        obj_im_dir, MAPPING[str(classes[idx])],
                        os.path.split(os.path.join(src_path, d))[1].replace(
                            ".jpg", "") + "_" + str(idx) + ".jpg"))
            except Exception as e:
                print(e)

    print("Number of objects: " + str(no_of_objects))


'''
Returns the list of cropped images based on the objects. The method make use of ground truth to crop the image.
'''
def crop_images_classwise_ground_truth(train_json_path, src_path, dest_path,
                                       category: str):
    if not os.path.exists(dest_path + '/obj_images'):
        os.makedirs(dest_path + '/obj_images')
    obj_im_dir = dest_path + '/obj_images'

    # MAPPING = {"text": 1, "title": 2, "list": 3, "table": 4, "figure": 5}
    MAPPING={vl:int(ky) for ky,vl in MAPPING.items()}
    # MAPPING = {'Date Block': 0, 'Logos': 1, 'Subject Block': 2, 'Body Block': 3, 'Circular ID': 4, 'Table': 5, 'Stamps/Seals': 6, 'Handwritten Text': 7, 'Copy-Forwarded To Block': 8, 'Address of Issuing Authority': 9, 'Signature': 10, 'Reference Block': 11, 'Signature Block': 12, 'Header Block': 13, 'Addressed To Block': 14}
    no_of_objects = 0
    with open(train_json_path) as f:
        data = json.load(f)
    annotations = data['annotations']
    file_names = os.listdir(src_path)
    file_ids = {
        x['id']: x['file_name']
        for x in data['images'] if x['file_name'] in file_names
    }
    for idx, d in tqdm(file_ids.items()):
        img = cv2.imread(os.path.join(src_path, d))
        if not os.path.exists(
                os.path.join(dest_path, 'obj_images', category)):
            os.makedirs(os.path.join(dest_path, 'obj_images', category))

        img = Image.open(os.path.join(src_path, d))
        boxes = [
            x['bbox'] for x in annotations if x['image_id'] == idx
            and x['category_id'] == MAPPING[category]
        ]
        for idx, box in enumerate(list(boxes)):
            no_of_objects += 1
            box = np.asarray(box, dtype=np.float32)

            crop_img = crop_object(img, box, True)
            crop_img.save(
                os.path.join(
                    obj_im_dir, category,
                    os.path.split(os.path.join(src_path, d))[1].replace(
                        ".jpg", "") + "_" + str(idx) + ".jpg"))

    print("Number of objects: " + str(no_of_objects))


def Random_wrapper(image_list, budget=10):
    rand_idx = np.random.permutation(len(image_list))[:budget]
    rand_idx = rand_idx.tolist()
    Random_results = [image_list[i] for i in rand_idx]

    return Random_results

def change_dir(image_results, src_dir, dest_dir):
    for image in image_results:
        source_img = image
        destination_img = os.path.join(dest_dir[0], os.path.basename(image))
        if not os.path.exists(dest_dir[0]) or not os.path.exists(dest_dir[1]):
            os.mkdir(dest_dir[0])
            os.mkdir(dest_dir[1])

        try:
            shutil.copy(source_img, destination_img)
        except shutil.SameFileError:
            print("Source and destination represents the same file.")

        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")

        # For other errors
        except Exception as e:
            print("Error occurred while copying file.", e)


        # removing the data from the lake data
        try:
            os.remove(source_img)
        except:
            pass

def create_labels_update(images, annotations, categories, filename):
    labels = {}
    labels['images'] = images
    labels['annotations'] = annotations
    labels['categories'] = categories

    with open(filename, "w") as f:
        json.dump(labels, f)


def remove_dir(dir_name):
    try:
        shutil.rmtree(dir_name)
    except:
        pass


def create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass

def get_original_images_path(subset_result:list,img_dir:str):
    return [os.path.join(img_dir,"_".join(os.path.basename(x).split("_")[:-1])) for x in subset_result]
    
def aug_train_subset(subset_result, train_data_json, lake_data_json, budget, src_dir, dest_dir):
    with open(lake_data_json, mode="r") as f:
        lake_dataset = json.load(f)
    with open(train_data_json, mode="r") as f:
        train_dataset = json.load(f)

    categories = lake_dataset['categories']
    image_list = list(filter(lambda x: x['file_name'] in subset_result, lake_dataset['images']))
    image_id = [image['id'] for image in image_list]
    annotations_shift = list(filter(lambda x: x['image_id'] in image_id, lake_dataset['annotations']))

    train_annotations = train_dataset['annotations'];
    train_image_list = train_dataset['images'];

    # appending the images to train images
    train_image_list += image_list;
    train_annotations += annotations_shift;

    #removing the images lake dataset.
    final_lake_image_list = list(filter(lambda x: x['file_name'] not in subset_result, lake_dataset['images']))
    image_id = [image['id'] for image in image_list]
    final_lake_annotations = list(filter(lambda x: x['image_id'] not in image_id, lake_dataset['annotations']))

    #moving data from lake set to train set.
    change_dir(subset_result, src_dir, dest_dir)

    #changing the coco-file for annotations
    create_labels_update(train_image_list, train_annotations, categories, train_data_json)
    create_labels_update(final_lake_image_list, final_lake_annotations, categories, lake_data_json)

def get_area(bbox):
  x=int(bbox[2])-int(bbox[0])
  y=int(bbox[3])-int(bbox[1])
  area=x*y
  return int(area)

def get_bounding_boxes(model, image_paths, image_id_mapping, annot_id):
    bounding_boxes = []

    for image_path in image_paths:
        try:
            print(image_path)
            image = cv2.imread(image_path)
            outputs = model(image)

            # Get the bounding boxes, labels, and scores
            instances = outputs["instances"]
            pred_boxes = instances.pred_boxes.tensor.tolist()
            pred_classes = instances.pred_classes.tolist()
            scores = instances.scores.tolist()

            # Print the predictions
            for i in range(len(pred_boxes)):
                # print(f"Bounding box: {pred_boxes[i]}, Label: {MAPPING[str(pred_classes[i])]}, Score: {scores[i]}")
                annot_id+=1
                bounding_boxes.append({
                    'iscrowd': 0,
                    'image_id': image_id_mapping[image_path],
                    'bbox': [int(i) for i in pred_boxes[i]],
                    'segmentation': [],
                    'category_id': int(pred_classes[i]),
                    'id': annot_id,
                    'area': get_area(pred_boxes[i])
                })
        except Exception as e:
            print(e)

    return bounding_boxes

def aug_train_subset_2(subset_result, train_data_json, model, budget, src_dir, dest_dir):
    print(subset_result)
    with open(train_data_json, mode="r") as f:
        train_dataset = json.load(f)

    categories = train_dataset['categories']
    max_image_id = max([image['id'] for image in train_dataset['images']])
    annot_id = max([annot['id'] for annot in train_dataset['annotations']])

    image_id_mapping = {image_name: idx + max_image_id+1 for idx, image_name in enumerate(subset_result)}

    # Update train image list with images from subset_result
    train_image_list = train_dataset['images'] + [
        {
            'id': image_id_mapping[image_path],
            'file_name': str(os.path.basename(image_path)),
            'height': int(cv2.imread(image_path).shape[0]),
            'width': int(cv2.imread(image_path).shape[1]),
        }
        for image_path in subset_result
    ]
    print('trian_image_list',train_image_list)

    # Get bounding box information using the model
    bounding_boxes = get_bounding_boxes(model, subset_result, image_id_mapping, annot_id)
    print('bounding_boxes :',bounding_boxes)
    
    # Update train annotations with bounding box information
    train_annotations = train_dataset['annotations'] + bounding_boxes

    # Remove the images from the source directory (change_dir function)
    change_dir(subset_result, src_dir, dest_dir)

    # Update the COCO file for train annotations
    create_labels_update(train_image_list, train_annotations, categories, train_data_json)
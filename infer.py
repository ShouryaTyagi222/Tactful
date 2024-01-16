from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg

import cv2
import argparse
import os

# from src.helper import *
# from src.configs import *

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


def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_path)
    cfg.MODEL.WEIGHTS=args.model_path
    predictor=create_model(cfg,'test')

    MAPPING = {'0': 'Date Block', '1': 'Logos', '2': 'Subject Block', '3': 'Body Block', '4': 'Circular ID', '5': 'Table', '6': 'Stamps/Seals', '7': 'Handwritten Text', '8': 'Copy-Forwarded To Block', '9': 'Address of Issuing Authority', '10': 'Signature', '11': 'Reference Block', '12': 'Signature Block', '13': 'Header Block', '14': 'Addressed To Block'}

    # Perform inference
    im = cv2.imread(args.img_path)
    outputs = predictor(im)

    # Get the bounding boxes, labels, and scores
    instances = outputs["instances"]
    pred_boxes = instances.pred_boxes.tensor.tolist()
    pred_classes = instances.pred_classes.tolist()
    scores = instances.scores.tolist()

    # Print the predictions
    for i in range(len(pred_boxes)):
        print(f"Bounding box: {pred_boxes[i]}, Label: {MAPPING[str(pred_classes[i])]}, Score: {scores[i]}")

    # Visualize the predictions
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('result.jpg',out.get_image()[:, :, ::-1])

def parse_args():
    parser = argparse.ArgumentParser(description="Infer Tactful", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--img_path", type=str, default=None, help="Path to the input image")
    parser.add_argument("-c", "--model_config_path", type=str, default=None, help="Path to the model config file")
    parser.add_argument("-m", "--model_path", type=str, default=None, help="Path to the model")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    main(arg)
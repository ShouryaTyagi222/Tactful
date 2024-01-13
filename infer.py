from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import argparse

from .src.helper import *
from .src.configs import *

def main(args):
    predictor=create_model(cfg,'test')

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
    cv2.imwrtie('result.jpg',out.get_image()[:, :, ::-1])

def parse_args():
    parser = argparse.ArgumentParser(description="Infer Tactful", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--img_path", type=str, default=None, help="Path to the input image")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    main(arg)
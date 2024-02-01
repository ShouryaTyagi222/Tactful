import os
import pandas as pd
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

from ..src.helper import *
from ..configs import *

logger = setup_logger(os.path.join(output_dir, cfg.TRAINING_NAME))

logger.info("Starting Initial_set Training")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
model = create_model(cfg)
torch.cuda.empty_cache()
model.train()
logger.info("Initial_set training complete")


iteration = args['iterations']
result_val = []
result_test = []

# del model
torch.cuda.empty_cache()

# step 2
# evaluate the inital model and get worst performing classcfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR , "model_final.pth")
model = create_model(cfg, "test")
result = do_evaluate(cfg, model, output_dir)
result_val.append(result['val_set'])
result_test.append(result['test_set'])
category_selection = []
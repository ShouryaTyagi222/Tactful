import os
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
import sys
sys.path.append("../")

from ..src.helper import *
from ..src.configs import *

# evaluate the inital model and get worst performing classcfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR , "model_final.pth")
model = create_model(cfg, "test")
result = do_evaluate(cfg, model, output_dir)
result_val.append(result['val_set'])
result_test.append(result['test_set'])
category_selection = []
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
import os
import torch

from src.helper import *

# basic args for tactful 
args = {
    "strategy":'random',      # strategy to be used for tactful
    "total_budget":150,  # Total data points available
    "budget":30,  # Budget per iteration
    "lake_size":150,  # Size of the lake dataset
    "train_size":46,  # Size of the training dataset
    "category":'Reference Block',   # Target Class     Note : use Stamps-Seals instead of Stamps/Seals due to path issues
    "device":1,
    "proposal_budget":30,  # Budget for proposal generation
    "iterations":5
}
args["output_path"] = args['strategy']

# mapping required for inference
MAPPING = {'0': 'Date Block', '1': 'Logos', '2': 'Subject Block', '3': 'Body Block', '4': 'Circular ID', '5': 'Table', '6': 'Stamps-Seals', '7': 'Handwritten Text', '8': 'Copy-Forwarded To Block', '9': 'Address of Issuing Authority', '10': 'Signature', '11': 'Reference Block', '12': 'Signature Block', '13': 'Header Block', '14': 'Addressed To Block'}

train_path = '/data/circulars/DATA/TACTFUL/faster_rcnn_output'    # path of the output dir

data_dir = '/data/circulars/DATA/TACTFUL/Data/random' # path to the data

train_data_dirs = (os.path.join(data_dir,"train"),
                   os.path.join(data_dir,"docvqa_train_coco.json"))
lake_data_dirs = (os.path.join(data_dir,"lake"),
                  os.path.join(data_dir,"docvqa_lake_coco.json"))
val_data_dirs = (os.path.join(data_dir,"val"),
                 os.path.join(data_dir,"docvqa_val_coco.json"))

# path to the query images dir
query_path = '/data/circulars/DATA/TACTFUL/Data/query_imgs'

# train a faster_rcnn model on the initial_set, add respective config file path
config_path = '/data/circulars/DATA/TACTFUL/Data/faster_rcnn_pub_config.yml'

training_name = args['output_path']
model_path = os.path.join(train_path, training_name)
if (not os.path.exists(model_path)):
    create_dir(model_path)
output_dir = os.path.join(model_path, "initial_training")

query_path = os.path.join(query_path, args['category'])

selection_arg = {"class":args['category'], 'eta':1, "model_path":model_path, 'smi_function':args['strategy']}


cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file(model_cfg))
cfg.merge_from_file(config_path)   #merge config file of the model
cfg.DATASETS.TRAIN = ("initial_set",)
cfg.DATASETS.TEST = ('val_set',)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15     # for the docvqa data
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 500
cfg.SOLVER.IMS_PER_BATCH = 10
cfg.MODEL.RPN.NMS_THRESH = 0.8
cfg.MODEL.RPN.POST_NMS_TOPK_TEST= 2000
# cfg.TEST.EVAL_PERIOD = 1000
cfg.OUTPUT_DIR = output_dir
cfg.TRAINING_NAME = training_name
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR , "model_final.pth")   # path to the model saved after initial model train saved in output dir

if torch.cuda.is_available():
    torch.cuda.set_device(args['device'])

#clearing data if already exist
remove_dataset("initial_set")
remove_dataset("val_set")

# Registering dataset intial_set for initial training, test_set and val_set for test and validation respectively.
register_coco_instances(
    "initial_set", {}, train_data_dirs[1], train_data_dirs[0])
register_coco_instances("val_set", {}, val_data_dirs[1], val_data_dirs[0])

iteration = args['iterations']
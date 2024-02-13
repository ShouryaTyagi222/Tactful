# Tactful

## config
`configs.py` consists of the required configs for the model training
- Tactful strategy - what tactful method to use
- MAPPING - Mapper for the classes
- total_budget - Total data points available
- lake size - Size of the lake dataset
- train_size - size fo the training dataset
- category - target class of the data for which the tactful is going to be used
- proposal_budget -Budget for proposal generation
- train_path - Path to the output dir where the model will be saved and validation results
- query_path - Path where query Dataset is available
- model_cfg - Provide the model_zoo model config for the required model.
- cfg - configs for the Required model
    1. NUM_WORKER - Number of Workers
    2. NUM_CLASSES - Number of Classes in the Data
    3. BASE_LR - Learning Rate
    4. IMS_PER_BATCH - Batch Size
- train/lake/test/val_data_dirs - Here provide the path to the img folder and coco file path for thhe required type of data.

## train
```
python train.py -i -m <INITIAL_MODEL_PATH>
```
## infer
```
python infer.py -i <INPUT_IMG_PATH> 
```

Note :
- the data is in Data folder
- to prepare data for training run coco_converter.py
- The outputs are saved in faster_rcnn_output.
- The val scores are in faster_rcnn_output/<strategy>/initial_training/val_scores.csv
- the Logs of the training are saved in faster_rcnn_output/<strategy>/initial_training/<strategy>log.txt
- Prepare the data using coco_converter.py before training everytime because of active learning the images are transfered from lake set to train set.


## Detectron2 Installation

- conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.0 -c pytorch
- conda install cython
- git clone https://github.com/facebookresearch/detectron2.git
- cd detectron2
- pip install -e .
- conda install pytorch torchvision torchaudio cudatoolkit=11.5 -c pytorch
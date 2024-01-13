# Tactful

## config
`src/configs.py` consists of the required configs for the model training
- Tactful strategy - what tactful method to use
- MAPPING - Mapper for the classes
- total_budget - Total data points available
- budget - Budget per iteration
- lake size - Size of the lake dataset
- train_size - size fo the training dataset
- category - what class of the data for which the tactful is going to be used
- proposal_budget -Budget for proposal generation
- iterations - Number of iterations (total_budget / budget)
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
python train.py -i <Yes if initial_training else Do not Specify >
```
## infer
```
python infer.py -i <INPUT_IMG_PATH> 
```
# python train.py -i -m /data/circulars/DATA/TACTFUL/Data/model_final.pth
import os
import pandas as pd
import torch
import argparse

from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances

from src.tactful_smi import TACTFUL_SMI
from src.helper import *
from configs import *

print('RUNNING')
def main():
    logger = setup_logger(os.path.join(output_dir, cfg.TRAINING_NAME))
    result_val = []
    iteration = args['iterations']
    selection_strag = args['strategy']
    selection_budget = args['budget']
    budget = args['total_budget']
    proposal_budget = args['proposal_budget']


    print('STRATEGY :', selection_strag)

    # Initial Training
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR , "model_final.pth")):
        logger.info("Starting Initial_set Training")
        try:
            cfg.MODEL.WEIGHTS = INITIAL_MODEL_PATH
            model = create_model(cfg)
            torch.cuda.empty_cache()
            model.train()
            logger.info("Initial_set training complete")

            torch.cuda.empty_cache()
        except Exception as e:
            logger.info(e)
            return
            
    i = 0
    
    while (i < iteration and budget > 0):
        # step 3
        # get embeddings for initial and lakeset from RESNET101

        if (selection_strag != "random"):

            # creating new query set for under performing class for each iteration
            remove_dir(os.path.join(model_path, "query_images"))
            try:
                os.remove(os.path.join(model_path, "data_query.csv"))
            except:
                pass

            # Cropping object based on ground truth for the query set.
            # The set is part of train set, so no need of using object detection model to find the bounding box.
            print('>>>',query_path)
            print('>>>',os.path.join(model_path,'query_images'))
            crop_images_classwise_ground_truth(train_data_dirs[1], query_path, os.path.join(
                model_path, "query_images"), args['category'])

            remove_dir(os.path.join(model_path, "lake_images"))
            try:
                os.remove(os.path.join(model_path, "data.csv"))
            except:
                pass
            model = create_model(cfg,'test')
            crop_images_classwise(
                model, lake_data_dirs[0], os.path.join(model_path, "lake_images"), proposal_budget=proposal_budget)

            selection_arg['iteration'] = i
            strategy_sel = TACTFUL_SMI(args = selection_arg)
            lake_image_list, subset_result = strategy_sel.select(proposal_budget)
            print('LENGTH OF SUBSET RESULT :',len(subset_result))
            subset_result = [lake_image_list[i] for i in subset_result]
            subset_result = list(
                set(get_original_images_path(subset_result,lake_data_dirs[0])))

        else:
            model = create_model(cfg,'test')
            lake_image_list = os.listdir(lake_data_dirs[0])
            subset_result = Random_wrapper(
                lake_image_list, selection_budget)

        # reducing the selection budget
        budget -= len(subset_result)
        if (budget > 0):

            # transferring images from lake set to train set
            aug_train_subset(subset_result, train_data_dirs[1], lake_data_dirs[1], budget, lake_data_dirs, train_data_dirs)
           
        # removing the old training images from the detectron configuration and adding new one
        remove_dataset("initial_set")
        register_coco_instances(
            "initial_set", {}, train_data_dirs[1], train_data_dirs[0])

        del model
        torch.cuda.empty_cache()
        # before starting the model active learning loop, calculating the embedding of the lake datset
        # change iteration as per the requirement
        cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"
        cfg.SOLVER.MAX_ITER = 500
        model = create_model(cfg, "train")
        model.train()

        # reevaluating the model train once again
        del model
        torch.cuda.empty_cache()
        # before starting the model active learning loop, calculating the embedding of the lake datset
        cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"
        model = create_model(cfg, "test")
        result = do_evaluate(cfg, model, output_dir)
        result_val.append(result['val_set'])
        # result_test.append(result['test_set'])

        # increasing the iteration number
        # publishing each iteration result to csv
        i += 1
        print("remaining_budget", budget)
        final_data = []
        temp = []
        for it in result_val:
            print(it)
            for k, val in it.items():
                temp = list(val.keys())
                final_data.append(list(val.values()))
        csv = pd.DataFrame(final_data, columns=temp)
        csv.to_csv(os.path.join(output_dir, '{}'.format(
            "val_scores"+selection_strag+".csv")))
        # final_data = []
        # for it in result_test:
        #     print(it)
        #     for k, val in it.items():
        #         temp = list(val.keys())
        #         final_data.append(list(val.values()))
        # csv = pd.DataFrame(final_data, columns=temp)
        # csv.to_csv(os.path.join(output_dir, '{}'.format(
        #     "test_scores"+selection_strag+".csv")))
    # except Exception as e:
    #     logger.error("Error while training:", e)

    # finally:
    final_data = []
    temp = []
    for i in result_val:
        print(i)
        for k, val in i.items():
            temp = list(val.keys())
            final_data.append(list(val.values()))
    csv = pd.DataFrame(final_data, columns=temp)
    csv.to_csv(os.path.join(output_dir, '{}'.format(
        "val_scores"+selection_strag+".csv")))
    # final_data = []
    # for i in result_test:
    #     print(i)
    #     for k, val in i.items():
    #         temp = list(val.keys())
    #         final_data.append(list(val.values()))
    # csv = pd.DataFrame(final_data, columns=temp)
    # csv.to_csv(os.path.join(output_dir, '{}'.format(
    #     "test_scores"+selection_strag+".csv")))

if __name__ == "__main__":
    main()
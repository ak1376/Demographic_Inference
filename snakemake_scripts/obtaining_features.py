# obtain_features.py

import numpy as np
import time
import pickle
import joblib
from sklearn.linear_model import LinearRegression
from preprocess import Processor, FeatureExtractor
from utils import process_and_save_data, visualizing_results, calculate_model_errors
import json

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def obtain_features(experiment_config, experiment_directory, num_sims_pretrain, num_sims_inference, normalization = False, remove_outliers = True):

    # print(f'Normalization option: {normalization}')

    # Load the experiment config
    with open(experiment_config, 'r') as f:
        experiment_config = json.load(f)
    
    # Load the experiment object to get the experiment directory
    with open(f'{experiment_directory}/experiment_obj.pkl', 'rb') as f:
        experiment_obj = pickle.load(f)
    experiment_directory = experiment_obj.experiment_directory


    processor = Processor(experiment_config, experiment_directory, recombination_rate=experiment_config["recombination_rate"], mutation_rate=experiment_config["mutation_rate"], window_length=experiment_config["window_length"])
    extractor = FeatureExtractor(experiment_directory, dadi_analysis=experiment_config["dadi_analysis"], moments_analysis=experiment_config["moments_analysis"], momentsLD_analysis=experiment_config["momentsLD_analysis"])

    all_indices = np.arange(num_sims_pretrain)
    np.random.shuffle(all_indices)
    n_train = int(0.8 * num_sims_pretrain)

    training_indices = all_indices[:n_train]
    validation_indices = all_indices[n_train:]
    testing_indices = np.arange(num_sims_inference)

    print(f'Number of Training Indices: {len(training_indices)}')
    print(f'Number of Validation Indices: {len(validation_indices)}')
    print(f'Number of Testing Indices: {len(testing_indices)}')

    for stage, indices in [
        ("training", training_indices),
        ("validation", validation_indices),
        ("testing", testing_indices),
    ]:
        result_ref = process_and_save_data(
            processor, indices, stage, experiment_directory
        )

        dadi_dict, moments_dict, momentsLD_dict = result_ref[0], result_ref[1], result_ref[2]

        # print(f'Dadi flag: {extractor.dadi_analysis}')
        # print(f'Moments flag: {extractor.moments_analysis}')
        # print(f'MomentsLD flag: {extractor.momentsLD_analysis}')
        
        if extractor.dadi_analysis:
            extractor.process_batch(dadi_dict, "dadi", stage, normalization=normalization)
        if extractor.moments_analysis:
            extractor.process_batch(moments_dict, "moments", stage, normalization=normalization)
       
        print(f'MomentsLD flag: {experiment_config["momentsLD_analysis"]}')
        if extractor.momentsLD_analysis:
            extractor.process_batch(momentsLD_dict, "momentsLD", stage, normalization=normalization)

    features, targets, feature_names = extractor.finalize_processing(remove_outliers=remove_outliers)
    
    preprocessing_results_obj = {
        stage: {
            "predictions": features[stage],
            "targets": targets[stage]
        } for stage in ["training", "validation", "testing"]
    }

    with open(f"{experiment_directory}/preprocessing_results_obj.pkl", "wb") as file:
        pickle.dump(preprocessing_results_obj, file)

    visualizing_results(
        preprocessing_results_obj,
        save_loc=experiment_directory,
        analysis="preprocessing_results",
        stages=["training", "validation"],
    )

    visualizing_results(
        preprocessing_results_obj,
        save_loc=experiment_directory,
        analysis="preprocessing_results_testing",
        stages=["testing"],
    )

    linear_mdl = LinearRegression()
    linear_mdl.fit(features["training"], targets["training"])

    linear_mdl_obj = {
        "model": linear_mdl,
        "training": {
            "predictions": linear_mdl.predict(features["training"]),
            "targets": targets["training"]
        },
        "validation": {
            "predictions": linear_mdl.predict(features["validation"]),
            "targets": targets["validation"]
        },
        "testing": {
            "targets": targets["testing"]
        }
    }

    print(f"{experiment_directory}/linear_mdl_obj.pkl")

    
    
    with open(f"{experiment_directory}/linear_mdl_obj.pkl", "wb") as file:
        pickle.dump(linear_mdl_obj, file)

    visualizing_results(
        linear_mdl_obj, "linear_results", save_loc=experiment_directory, stages=["training", "validation"]
    )

    preprocessing_errors = calculate_model_errors(
        preprocessing_results_obj, "preprocessing", datasets=["training", "validation"]
    )
    linear_errors = calculate_model_errors(linear_mdl_obj, "linear", datasets=["training", "validation"])

    all_errors = {**preprocessing_errors, **linear_errors}

    with open(f"{experiment_directory}/preprocessing_data_error.txt", "w") as f:
        for model, datasets in all_errors.items():
            f.write(f"\n{model.upper()} Model Errors:\n")
            for dataset, error in datasets.items():
                f.write(f"  {dataset.capitalize()} RMSE: {error:.4f}\n")

    joblib.dump(linear_mdl, f"{experiment_directory}/linear_regression_model.pkl")

    print("Feature extraction and model training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_config', type=str, required=True)
    parser.add_argument('--experiment_directory', type=str, required=True)
    parser.add_argument('--num_sims_pretrain', type=int, required=True)
    parser.add_argument('--num_sims_inference', type=int, required=True)
    parser.add_argument('--normalization', type=str2bool, default=True)
    parser.add_argument('--remove_outliers', type=str2bool, default=True)
    args = parser.parse_args()

    obtain_features(
        args.experiment_config,
        args.experiment_directory,
        args.num_sims_pretrain,
        args.num_sims_inference,
        args.normalization,
        args.remove_outliers
    )
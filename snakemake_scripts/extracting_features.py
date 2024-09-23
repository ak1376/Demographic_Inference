import pickle
import json

def getting_the_features(postprocessing_results_filepath, sim_directory):

    with open(postprocessing_results_filepath, "rb") as file:
        postprocessing_results_obj = pickle.load(file)

    # # print(f"TRAINING FEATURES SHAPE: {preprocessing_results_obj['training']['predictions'].shape}")
    # # print(f"TRAINING TARGETS SHAPE: {preprocessing_results_obj['training']['targets'].shape}")
    # # print(f"VALIDATION FEATURES SHAPE: {preprocessing_results_obj['validation']['predictions'].shape}")
    # # print(f"VALIDATION TARGETS SHAPE: {preprocessing_results_obj['validation']['targets'].shape}")

    # training_features = preprocessing_results_obj["training"]["predictions"]
    # training_targets = preprocessing_results_obj["training"]["targets"]
    # validation_features = preprocessing_results_obj["validation"]["predictions"]
    # validation_targets = preprocessing_results_obj["validation"]["targets"]

    # testing_features = preprocessing_results_obj["testing"]["predictions"]
    # testing_targets = preprocessing_results_obj["testing"]["targets"]

    # # Needs to be some flag checking if this is true or not. 
    # additional_features = None
    # if config["use_FIM"]:
    #     additional_features = {}
    #     additional_features['training'] = preprocessing_results_obj['training']['upper_triangular_FIM']
    #     additional_features['validation'] = preprocessing_results_obj['validation']['upper_triangular_FIM']
    #     additional_features['testing'] = preprocessing_results_obj['testing']['upper_triangular_FIM']

    # I want to save a dictionary of training, validation, and testing features and targets.

    features = {
        "training": {"features": postprocessing_results_obj['training']['predictions'], "targets": postprocessing_results_obj['training']['targets']},
        "validation": {"features": postprocessing_results_obj['validation']['predictions'], "targets": postprocessing_results_obj['validation']['targets']},
    }

    print(f'Training features shape: {features["training"]["features"].shape}')
    print(f'Validation features shape: {features["validation"]["features"].shape}')

    print(f'Training targets shape: {features["training"]["targets"].shape}')
    print(f'Validation targets shape: {features["validation"]["targets"].shape}')

    # Now save the dictionary as a pickle
    with open(f"{sim_directory}/features_and_targets.pkl", "wb") as file:
        pickle.dump(features, file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocessing_results_filepath", type=str, required=True)
    parser.add_argument("--sim_directory", type=str, required=True)
    args = parser.parse_args()

    getting_the_features(args.postprocessing_results_filepath, args.sim_directory)

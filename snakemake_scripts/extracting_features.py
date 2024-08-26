import pickle

def getting_the_features(preprocessing_results_filepath, experiment_directory):
    with open(preprocessing_results_filepath, "rb") as file:
        preprocessing_results_obj = pickle.load(file)

    training_features = preprocessing_results_obj["training"]["predictions"]
    training_targets = preprocessing_results_obj["training"]["targets"]
    validation_features = preprocessing_results_obj["validation"]["predictions"]
    validation_targets = preprocessing_results_obj["validation"]["targets"]

    testing_features = preprocessing_results_obj["testing"]["predictions"]
    testing_targets = preprocessing_results_obj["testing"]["targets"]

    
    # I want to save a dictionary of training, validation, and testing features and targets.
    features = {
        "training_features": training_features,
        "validation_features": validation_features,
        "testing_features": testing_features,
        "training_targets": training_targets,
        "validation_targets": validation_targets,
        "testing_targets": testing_targets
    }

    # Now save the dictionary as a pickle
    with open(f"{experiment_directory}/features_and_targets.pkl", "wb") as file:
        pickle.dump(features, file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing_results_filepath', type=str, required=True)
    parser.add_argument('--experiment_directory', type=str, required=True)
    args = parser.parse_args()


    getting_the_features(args.preprocessing_results_filepath, args.experiment_directory)
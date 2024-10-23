import pickle

def getting_the_features(postprocessing_results_filepath, sim_directory):

    with open(postprocessing_results_filepath, "rb") as file:
        postprocessing_results_obj = pickle.load(file)

    print(postprocessing_results_obj.keys())

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

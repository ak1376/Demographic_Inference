# from src.models import XGBoost
# import argparse



# def main():

#     # TODO: Have a separate config file for xgboost hyperparameters
#     xgb_model = XGBoost(
#         objective="reg:squarederror",
#         n_estimators=100,
#         learning_rate=0.1,
#         max_depth=5,
#         verbosity=2,
#         train_percentage=0.8,
#         loo=None
#     )


#     pass






# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--experiment_directory", type=str, required=True)
#     parser.add_argument("--model_config_file", type=str, required=True)
#     parser.add_argument("--features_file", type=str, required=True)
#     parser.add_argument("--color_shades", type=str, required=True)
#     parser.add_argument("--main_colors", type=str, required=True)
#     args = parser.parse_args()

#     main(
#         args.experiment_directory,
#         args.model_config_file,
#         args.features_file,
#         args.color_shades,
#         args.main_colors,
#     )

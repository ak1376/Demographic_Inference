import os
import json

# Load the configuration file
# configfile: "experiment_config.yaml"
# experiment_config = config["experiment"]
# model_config = config["model"]

CONFIG_FILEPATH = '/home/akapoor/kernlab/Demographic_Inference/experiment_config.json'
MODEL_CONFIG_FILEPATH = '/home/akapoor/kernlab/Demographic_Inference/model_config.json'
MODEL_CONFIG_XGBOOST_FILEPATH = '/home/akapoor/kernlab/Demographic_Inference/model_config_xgb.json'


with open(CONFIG_FILEPATH, 'r') as f:
   experiment_config = json.load(f)

with open(MODEL_CONFIG_FILEPATH, 'r') as f:
   model_config = json.load(f)


CWD = os.getcwd()

# Use double quotes for the dictionary keys inside the f-string
EXPERIMENT_DIRECTORY = f'{experiment_config['demographic_model']}_dadi_analysis_{experiment_config['dadi_analysis']}_moments_analysis_{experiment_config['moments_analysis']}_momentsLD_analysis_{experiment_config['momentsLD_analysis']}_seed_{experiment_config['seed']}'
EXPERIMENT_NAME = f'sims_pretrain_{experiment_config["num_sims_pretrain"]}_sims_inference_{experiment_config["num_sims_inference"]}_seed_{experiment_config["seed"]}_num_replicates_{experiment_config["k"]}_top_values_{experiment_config["top_values_k"]}'
SIM_DIRECTORY = f"{EXPERIMENT_DIRECTORY}/sims/{EXPERIMENT_NAME}"

# Check if hidden_size is a list, and if so, join the elements with "_"
hidden_size = model_config['neural_net_hyperparameters']['hidden_size']
if isinstance(hidden_size, list):
    hidden_size_str = "_".join(map(str, hidden_size))  # Join list elements with "_"
else:
    hidden_size_str = str(hidden_size)  # Convert integer to string if not a list

# Build the MODEL_DIRECTORY string
MODEL_DIRECTORY = (
    f"{EXPERIMENT_DIRECTORY}/models/{EXPERIMENT_NAME}/"
    f"num_hidden_neurons_{hidden_size_str}_"
    f"num_hidden_layers_{model_config['neural_net_hyperparameters']['num_layers']}_"
    f"num_epochs_{model_config['neural_net_hyperparameters']['num_epochs']}_"
    f"dropout_value_{model_config['neural_net_hyperparameters']['dropout_rate']}_"
    f"weight_decay_{model_config['neural_net_hyperparameters']['weight_decay']}_"
    f"batch_size_{model_config['neural_net_hyperparameters']['batch_size']}_"
    f"EarlyStopping_{model_config['neural_net_hyperparameters']['EarlyStopping']}"
)

# Set working directory
workdir: "/projects/kernlab/akapoor/Demographic_Inference"

# Add wildcard constraints
wildcard_constraints:
    sim_number=r"\d+",    # Note the 'r' prefix
    window_number=r"\d+"  # Note the 'r' prefix

rule all:
    input:
        f"{SIM_DIRECTORY}/postprocessing_results.pkl",
        f"{SIM_DIRECTORY}/features_and_targets.pkl",
        # f"{MODEL_DIRECTORY}/linear_regression_model.pkl",
        # f"{MODEL_DIRECTORY}/snn_results.pkl",
        # f"{MODEL_DIRECTORY}/snn_model.pth"
        # f"{MODEL_DIRECTORY}/xgb_model_obj.pkl",
        # f"{MODEL_DIRECTORY}/inferred_params_GHIST_bottleneck.txt"

# Rule: Create experiment
rule create_experiment:
    params:
        CONFIG_FILEPATH = CONFIG_FILEPATH,
        EXPERIMENT_NAME = EXPERIMENT_NAME,
        EXPERIMENT_DIRECTORY = EXPERIMENT_DIRECTORY,
        SIM_DIRECTORY = SIM_DIRECTORY,
    output:
        config_file = f"{SIM_DIRECTORY}/config.json",
        inference_config_file = f"{SIM_DIRECTORY}/inference_config_file.json",
        colors_shades_file = f"{SIM_DIRECTORY}/color_shades.pkl",
        main_colors_file = f"{SIM_DIRECTORY}/main_colors.pkl"
    run:
        with open(output.config_file, 'w') as f:
            json.dump(experiment_config, f, indent=4)

        shell(f"""
            PYTHONPATH={CWD} python {CWD}/snakemake_scripts/setup_experiment.py \
            --config_file {params.CONFIG_FILEPATH} \
            --experiment_name {params.EXPERIMENT_NAME} \
            --experiment_directory {params.EXPERIMENT_DIRECTORY} \
            --sim_directory {params.SIM_DIRECTORY} \
        """)

rule run_simulation:
    params:
        CONFIG_FILEPATH = CONFIG_FILEPATH,
        SIM_DIRECTORY = '/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences'
    output:
        sampled_params_pkl = "/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences/simulation_results/sampled_params_{sim_number}.pkl",
        metadata_file = "/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences/simulation_results/sampled_params_metadata_{sim_number}.txt",
        sfs_file = "/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences/simulation_results/SFS_sim_{sim_number}.pkl",
        tree_sequence_file = "/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences/simulation_results/ts_sim_{sim_number}.trees"
    shell:
        """
        PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference/ python /projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/single_simulation.py \
        --experiment_config {params.CONFIG_FILEPATH} \
        --sim_directory {params.SIM_DIRECTORY} \
        --sim_number {wildcards.sim_number}
        """

rule genome_windows:
    input:
        tree_sequence_file = rules.run_simulation.output.tree_sequence_file
    output:
        samples_file = "/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}/window_{window_number}/samples.txt",
        flat_map_file = "/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}/window_{window_number}/flat_map.txt",
        metadata_file = "/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}/window_{window_number}/individual_file_metadata.txt",
        vcf_file = "/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}/window_{window_number}/window.{window_number}.vcf.gz"  # Add this line

    params:
        window_number = lambda wildcards: wildcards.window_number,
        CONFIG_FILEPATH = CONFIG_FILEPATH
    shell:
        """
        mkdir -p /projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{wildcards.sim_number}
        PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference python /projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/obtain_genome_vcfs.py \
        --tree_sequence_file {input.tree_sequence_file} \
        --experiment_config_filepath {params.CONFIG_FILEPATH} \
        --genome_sim_directory sampled_genome_windows \
        --window_number {wildcards.window_number} \
        --sim_number {wildcards.sim_number}
        """

rule combine_metadata:
    input:
        tree_sequence_file = rules.run_simulation.output.tree_sequence_file,
        experiment_config = CONFIG_FILEPATH,
        samples_files = lambda wildcards: [
            f"/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{wildcards.sim_number}/window_{i}/samples.txt"
            for i in range(json.load(open(CONFIG_FILEPATH))["num_windows"])
        ]
    output:
        metadata_file = "/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}/metadata.txt"
    params:
        num_windows = lambda wildcards: json.load(open(CONFIG_FILEPATH))["num_windows"]
    shell:
        """
        # Create output directory
        mkdir -p $(dirname {output.metadata_file})
        
        # Concatenate all VCF file paths into the metadata file
        for ((i=0; i<{params.num_windows}; i++)); do
            echo "/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{wildcards.sim_number}/window_$i/window.$i.vcf.gz" >> {output.metadata_file}
        done
        """

rule calculate_LD_stats:
    input:
        pop_file_path = "/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}/window_{window_number}/samples.txt",
        flat_map_file = "/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}/window_{window_number}/flat_map.txt",
        metadata_file = "/projects/kernlab/akapoor/Demographic_Inference/sampled_genome_windows/sim_{sim_number}/metadata.txt",
        sampled_params_pkl = "/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences/simulation_results/sampled_params_{sim_number}.pkl"
    output:
        processed_file = "/projects/kernlab/akapoor/Demographic_Inference/LD_inferences/sim_{sim_number}/window_{window_number}/ld_stats_window.{window_number}.pkl"
    params:
        window_number = lambda wildcards: wildcards.window_number
    shell:
        """
        echo "Extracting VCF file for window number {wildcards.window_number}"
        vcf_filepath=$(sed -n "$(( {wildcards.window_number} + 1 ))p" {input.metadata_file})
        echo "Processing VCF file: $vcf_filepath"
        PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference/ python /projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/ld_stats.py \
            --vcf_filepath "$vcf_filepath" \
            --pop_file_path {input.pop_file_path} \
            --flat_map_path {input.flat_map_file} \
            --sim_directory /projects/kernlab/akapoor/Demographic_Inference/LD_inferences \
            --sim_number {wildcards.sim_number} \
            --window_number {wildcards.window_number}
        """

rule gather_ld_stats:
    input:
        ld_stats_files=lambda wildcards: expand(
            "/projects/kernlab/akapoor/Demographic_Inference/LD_inferences/sim_{sim_number}/window_{window_number}/ld_stats_window.{window_number}.pkl",
            sim_number=wildcards.sim_number,
            window_number=range(0, experiment_config['num_windows'])
        )
    output:
        combined_ld_stats_sim="/projects/kernlab/akapoor/Demographic_Inference/combined_LD_inferences/sim_{sim_number}/combined_LD_stats_sim_{sim_number}.pkl"
    shell:
        """
        mkdir -p /projects/kernlab/akapoor/Demographic_Inference/combined_LD_inferences/sim_{wildcards.sim_number}

        # Use process substitution to pass newline-separated filenames
        PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference python /projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/combine_ld_stats.py \
            --ld_stats_files <(printf "%s\\n" {input.ld_stats_files}) \
            --sim_number {wildcards.sim_number}
        """

rule obtain_MomentsLD_feature:
    input:
        combined_ld_stats_path = rules.gather_ld_stats.output.combined_ld_stats_sim,
        sampled_params_pkl = rules.run_simulation.output.sampled_params_pkl
    output:
        software_inferences = "/projects/kernlab/akapoor/Demographic_Inference/final_LD_inferences/momentsLD_inferences_sim_{sim_number}.pkl"
    shell:
        """
        mkdir -p /projects/kernlab/akapoor/Demographic_Inference/final_LD_inferences
        PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference/ python /projects/kernlab/akapoor/Demographic_Inference//snakemake_scripts/momentsLD_analysis.py \
        --combined_ld_stats_path {input.combined_ld_stats_path} \
        --sampled_params {input.sampled_params_pkl} \
        --experiment_config_filepath {CONFIG_FILEPATH} \
        --sim_directory /projects/kernlab/akapoor/Demographic_Inference/final_LD_inferences \
        --sim_number {wildcards.sim_number}
        """

rule obtain_feature:
    input:
        sampled_params_pkl = rules.run_simulation.output.sampled_params_pkl,
        SFS = rules.run_simulation.output.sfs_file
    output:
        software_inferences = "/projects/kernlab/akapoor/Demographic_Inference/moments_dadi_features/sim_{sim_number}/{analysis}/replicate_{replicate_number}/replicate_{replicate_number}.pkl"
    params:
        sim_directory = "/projects/kernlab/akapoor/Demographic_Inference/moments_dadi_features",
        analysis = lambda wildcards: "dadi" if wildcards.analysis == "dadi" else "moments"
    shell:
        """
        PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference/ python /projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/obtain_single_feature.py \
        --sfs_file {input.SFS} \
        --sampled_params_pkl {input.sampled_params_pkl} \
        --experiment_config_filepath {CONFIG_FILEPATH} \
        --sim_directory {params.sim_directory} \
        --sim_number {wildcards.sim_number} \
        --replicate_number {wildcards.replicate_number}
        """


rule aggregate_top_k_results:
    input:
        dadi_files = expand("/projects/kernlab/akapoor/Demographic_Inference/moments_dadi_features/sim_{{sim_number}}/dadi/replicate_{replicate}/replicate_{replicate}.pkl", 
                          replicate=range(json.load(open(CONFIG_FILEPATH))["k"])),
        moments_files = expand("/projects/kernlab/akapoor/Demographic_Inference/moments_dadi_features/sim_{{sim_number}}/moments/replicate_{replicate}/replicate_{replicate}.pkl", 
                             replicate=range(json.load(open(CONFIG_FILEPATH))["k"])),
        sfs_file = "/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences/simulation_results/SFS_sim_{sim_number}.pkl",
        params_file = "/projects/kernlab/akapoor/Demographic_Inference/simulated_parameters_and_inferences/simulation_results/sampled_params_{sim_number}.pkl"
    output:
        "/projects/kernlab/akapoor/Demographic_Inference/moments_dadi_features/software_inferences_sim_{sim_number}.pkl"
    params:
        top_k = json.load(open(CONFIG_FILEPATH))["top_values_k"]
    shell:
        """
        PYTHONPATH=/projects/kernlab/akapoor/Demographic_Inference/ python /projects/kernlab/akapoor/Demographic_Inference/snakemake_scripts/aggregate_top_k.py \
        --dadi_files {input.dadi_files} \
        --moments_files {input.moments_files} \
        --sfs_file {input.sfs_file} \
        --params_file {input.params_file} \
        --top_k {params.top_k} \
        --sim_number {wildcards.sim_number} \
        """

# Rule to gather software_inferences
def gather_software_inferences(wildcards):
    return expand(
        f"/projects/kernlab/akapoor/Demographic_Inference/moments_dadi_features/software_inferences_sim_{{sim_number}}.pkl",
        sim_number=range(0, experiment_config['num_sims_pretrain'])
    )

# Rule to gather momentsLD_inferences
def gather_momentsLD_inferences(wildcards):
    return expand(
        f"/projects/kernlab/akapoor/Demographic_Inference/final_LD_inferences/momentsLD_inferences_sim_{{sim_number}}.pkl",
        sim_number=range(0, experiment_config['num_sims_pretrain'])
    )

rule aggregate_features:
    input:
        software_inferences = gather_software_inferences,
        momentsLD_inferences = gather_momentsLD_inferences,
        experiment_config_filepath = CONFIG_FILEPATH
    output:
        preprocessing_results = f"{SIM_DIRECTORY}/preprocessing_results_obj.pkl",
        training_features = f"{SIM_DIRECTORY}/training_features.npy",
        training_targets = f"{SIM_DIRECTORY}/training_targets.npy",
        validation_features = f"{SIM_DIRECTORY}/validation_features.npy",
        validation_targets = f"{SIM_DIRECTORY}/validation_targets.npy"
    params:
        sim_directory = SIM_DIRECTORY
    shell:
        """
        mkdir -p {params.sim_directory}
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/aggregate_all_features.py \
            {input.experiment_config_filepath} \
            {params.sim_directory} \
            --software_inferences_dir {input.software_inferences:q} \
            --MomentsLD_inferences_dir {input.momentsLD_inferences:q}
        """


rule postprocessing:
    input:
        experiment_config_filepath = CONFIG_FILEPATH,
        training_features_filepath = rules.aggregate_features.output.training_features,
        validation_features_filepath = rules.aggregate_features.output.validation_features,
        training_targets_filepath = rules.aggregate_features.output.training_targets,
        validation_targets_filepath = rules.aggregate_features.output.validation_targets
    params:
        SIM_DIRECTORY = SIM_DIRECTORY
    output:
        postprocessing_results = f'{SIM_DIRECTORY}/postprocessing_results.pkl'
    shell:
        """
        mkdir -p {params.SIM_DIRECTORY}
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/postprocessing.py \
        --config_file {input.experiment_config_filepath} \
        --training_features_filepath {input.training_features_filepath} \
        --validation_features_filepath {input.validation_features_filepath} \
        --training_targets_filepath {input.training_targets_filepath} \
        --validation_targets_filepath {input.validation_targets_filepath} \
        --sim_directory {params.SIM_DIRECTORY}
        """

rule get_features:
    input:
        postprocessing_results = rules.postprocessing.output.postprocessing_results,
    output:
        features_output = f"{SIM_DIRECTORY}/features_and_targets.pkl",

    params:
        sim_directory = SIM_DIRECTORY

    # conda: 
    #     "myenv"
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/extracting_features.py \
        --postprocessing_results_filepath {input.postprocessing_results} \
        --sim_directory {params.sim_directory} \
        """

rule linear_evaluation: 
    input:
        # TEMPORARY
    #    features_and_targets_path = '/sietch_colab/akapoor/Demographic_Inference/bottleneck_experiments_seed_42/sims/sims_pretrain_10000_sims_inference_5_seed_42_num_replicates_5/features_and_targets.pkl',
    #    experiment_config_filepath = CONFIG_FILEPATH,
    #    color_shades_file = '/sietch_colab/akapoor/Demographic_Inference/bottleneck_experiments_seed_42/sims/sims_pretrain_10000_sims_inference_5_seed_42_num_replicates_5/color_shades.pkl',
    #    main_colors_file = '/sietch_colab/akapoor/Demographic_Inference/bottleneck_experiments_seed_42/sims/sims_pretrain_10000_sims_inference_5_seed_42_num_replicates_5/main_colors.pkl'
        features_and_targets_path = rules.get_features.output.features_output,
        experiment_config_filepath = CONFIG_FILEPATH,
        color_shades_file = rules.create_experiment.output.colors_shades_file,
        main_colors_file = rules.create_experiment.output.main_colors_file
    
    output:
        linear_model = f"{MODEL_DIRECTORY}/linear_regression_model.pkl"  
    
    params:
        model_directory = MODEL_DIRECTORY

    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/linear_evaluation.py \
        --features_and_targets_filepath {input.features_and_targets_path} \
        --experiment_config_filepath {input.experiment_config_filepath} \
        --color_shades_file {input.color_shades_file} \
        --main_colors_file {input.main_colors_file} \
        --model_directory {params.model_directory}
        """

# rule xgboost_evaluation: 
#     input:
#         features_file = rules.get_features.output.features_output,
#         color_shades_file = rules.create_experiment.output.colors_shades_file,
#         main_colors_file = rules.create_experiment.output.main_colors_file
    
#     params:
#         experiment_directory = MODEL_DIRECTORY,
#         MODEL_CONFIG_XGBOOST_FILEPATH = MODEL_CONFIG_XGBOOST_FILEPATH  

#     output:
#         xgboost_model = f"{MODEL_DIRECTORY}/xgb_model_obj.pkl"

#     shell:
#         """
#         PYTHONPATH={CWD} python {CWD}/snakemake_scripts/setup_xgboost.py \
#         --model_config {params.MODEL_CONFIG_XGBOOST_FILEPATH} \
#         --experiment_directory {params.experiment_directory} \
#         --features_file {input.features_file} \
#         --color_shades {input.color_shades_file} \
#         --main_colors {input.main_colors_file}
#         """


rule train_and_predict:
    input:
        features_file = rules.get_features.output.features_output,
        color_shades_file = rules.create_experiment.output.colors_shades_file,
        main_colors_file = rules.create_experiment.output.main_colors_file,
    params:
        experiment_directory = MODEL_DIRECTORY,
        MODEL_CONFIG_FILEPATH = MODEL_CONFIG_FILEPATH  # Add this if missing
    output:
        model_results = f"{MODEL_DIRECTORY}/snn_results.pkl",
        trained_model = f"{MODEL_DIRECTORY}/snn_model.pth"
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/setup_trainer.py \
        --model_config {params.MODEL_CONFIG_FILEPATH} \
        --experiment_directory {params.experiment_directory} \
        --features_file {input.features_file} \
        --color_shades {input.color_shades_file} \
        --main_colors {input.main_colors_file}
        """

# rule get_inferred_params:
#     input:
#         model_config = MODEL_CONFIG_FILEPATH,
#         config = CONFIG_FILEPATH,
#         trained_weights = rules.train_and_predict.output.trained_model,

#     params:
#         experiment_directory = MODEL_DIRECTORY,
#         inference_obj_path = f"{MODEL_DIRECTORY}/inference_results_obj.pkl"


#     output:
#         f"{MODEL_DIRECTORY}/inference_results_obj.pkl",
#         f"{MODEL_DIRECTORY}/inferred_params_GHIST_bottleneck.txt"
#     shell:
#         """
#         PYTHONPATH={CWD} python {CWD}/snakemake_scripts/inference_snakemake.py \
#         --model_config {input.model_config} \
#         --config {input.config} \
#         --trained_weights {input.trained_weights} \
#         --inference_obj_path {params.inference_obj_path} \
#         --experiment_directory {params.experiment_directory}
#         """



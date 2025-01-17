import os
import json

# Define the base directory for the project
BASE_DIR = '/sietch_colab/akapoor/Demographic_Inference'

# Define file paths relative to BASE_DIR
CONFIG_FILEPATH = os.path.join(BASE_DIR, 'experiment_config.json')
MODEL_CONFIG_FILEPATH = os.path.join(BASE_DIR, 'model_config.json')
MODEL_CONFIG_XGBOOST_FILEPATH = os.path.join(BASE_DIR, 'model_config_xgb.json')

# Load configuration files
with open(CONFIG_FILEPATH, 'r') as f:
    experiment_config = json.load(f)

with open(MODEL_CONFIG_FILEPATH, 'r') as f:
    model_config = json.load(f)

# Get the current working directory
CWD = os.getcwd()

# Define experiment and model directories
EXPERIMENT_DIRECTORY = (
    f"{experiment_config['demographic_model']}_dadi_analysis_"
    f"{experiment_config['dadi_analysis']}_moments_analysis_"
    f"{experiment_config['moments_analysis']}_momentsLD_analysis_"
    f"{experiment_config['momentsLD_analysis']}_seed_{experiment_config['seed']}"
)
EXPERIMENT_NAME = (
    f"sims_pretrain_{experiment_config['num_sims_pretrain']}_"
    f"sims_inference_{experiment_config['num_sims_inference']}_"
    f"seed_{experiment_config['seed']}_num_replicates_{experiment_config['k']}_"
    f"top_values_{experiment_config['top_values_k']}"
)
SIM_DIRECTORY = os.path.join(BASE_DIR, f"{EXPERIMENT_DIRECTORY}/sims/{EXPERIMENT_NAME}")

# Check if hidden_size is a list, and join the elements if so
hidden_size = model_config['neural_net_hyperparameters']['hidden_size']
hidden_size_str = "_".join(map(str, hidden_size)) if isinstance(hidden_size, list) else str(hidden_size)

MODEL_DIRECTORY = os.path.join(
    BASE_DIR,
    f"{EXPERIMENT_DIRECTORY}/models/{EXPERIMENT_NAME}/"
    f"num_hidden_neurons_{hidden_size_str}_"
    f"num_hidden_layers_{model_config['neural_net_hyperparameters']['num_layers']}_"
    f"num_epochs_{model_config['neural_net_hyperparameters']['num_epochs']}_"
    f"dropout_value_{model_config['neural_net_hyperparameters']['dropout_rate']}_"
    f"weight_decay_{model_config['neural_net_hyperparameters']['weight_decay']}_"
    f"batch_size_{model_config['neural_net_hyperparameters']['batch_size']}_"
    f"EarlyStopping_{model_config['neural_net_hyperparameters']['EarlyStopping']}"
)

# Set the working directory
workdir: BASE_DIR

# Wildcard constraints
wildcard_constraints:
    sim_number=r"\d+",
    window_number=r"\d+"

# Rule: All
rule all:
    input:
        os.path.join(SIM_DIRECTORY, "postprocessing_results.pkl"),
        os.path.join(SIM_DIRECTORY, "features_and_targets.pkl")
        # os.path.join(MODEL_DIRECTORY, "linear_regression_model.pkl"),
        # os.path.join(MODEL_DIRECTORY, "snn_results.pkl"),
        # os.path.join(MODEL_DIRECTORY, "snn_model.pth")

# Rule: Create experiment
rule create_experiment:
    params:
        CONFIG_FILEPATH=CONFIG_FILEPATH,
        EXPERIMENT_NAME=EXPERIMENT_NAME,
        EXPERIMENT_DIRECTORY=EXPERIMENT_DIRECTORY,
        SIM_DIRECTORY=SIM_DIRECTORY
    output:
        config_file=os.path.join(SIM_DIRECTORY, "config.json"),
        inference_config_file=os.path.join(SIM_DIRECTORY, "inference_config_file.json"),
        colors_shades_file=os.path.join(SIM_DIRECTORY, "color_shades.pkl"),
        main_colors_file=os.path.join(SIM_DIRECTORY, "main_colors.pkl")
    run:
        with open(output.config_file, 'w') as f:
            json.dump(experiment_config, f, indent=4)

        shell(f"""
            PYTHONPATH={CWD} python {CWD}/snakemake_scripts/setup_experiment.py \
            --config_file {params.CONFIG_FILEPATH} \
            --experiment_name {params.EXPERIMENT_NAME} \
            --experiment_directory {params.EXPERIMENT_DIRECTORY} \
            --sim_directory {params.SIM_DIRECTORY}
        """)

# Rule: Run simulation
rule run_simulation:
    params:
        CONFIG_FILEPATH=CONFIG_FILEPATH,
        SIM_DIRECTORY=os.path.join(BASE_DIR, "simulated_parameters_and_inferences")
    output:
        sampled_params_pkl=os.path.join(BASE_DIR, "simulated_parameters_and_inferences/simulation_results/sampled_params_{sim_number}.pkl"),
        metadata_file=os.path.join(BASE_DIR, "simulated_parameters_and_inferences/simulation_results/sampled_params_metadata_{sim_number}.txt"),
        sfs_file=os.path.join(BASE_DIR, "simulated_parameters_and_inferences/simulation_results/SFS_sim_{sim_number}.pkl"),
        tree_sequence_file=os.path.join(BASE_DIR, "simulated_parameters_and_inferences/simulation_results/ts_sim_{sim_number}.trees")
    shell:
        """
        PYTHONPATH={BASE_DIR} python {BASE_DIR}/snakemake_scripts/single_simulation.py \
        --experiment_config {params.CONFIG_FILEPATH} \
        --sim_directory {params.SIM_DIRECTORY} \
        --sim_number {wildcards.sim_number}
        """

# Rule: Genome windows
rule genome_windows:
    input:
        tree_sequence_file=rules.run_simulation.output.tree_sequence_file
    output:
        samples_file=f"{BASE_DIR}/sampled_genome_windows/sim_{{sim_number}}/window_{{window_number}}/samples.txt",
        flat_map_file=f"{BASE_DIR}/sampled_genome_windows/sim_{{sim_number}}/window_{{window_number}}/flat_map.txt",
        metadata_file=f"{BASE_DIR}/sampled_genome_windows/sim_{{sim_number}}/window_{{window_number}}/individual_file_metadata.txt",
        vcf_file=f"{BASE_DIR}/sampled_genome_windows/sim_{{sim_number}}/window_{{window_number}}/window.{{window_number}}.vcf.gz"
    params:
        window_number=lambda wildcards: wildcards.window_number,
        CONFIG_FILEPATH=CONFIG_FILEPATH
    shell:
        """
        mkdir -p {BASE_DIR}/sampled_genome_windows/sim_{wildcards.sim_number}
        PYTHONPATH={BASE_DIR} python {BASE_DIR}/snakemake_scripts/obtain_genome_vcfs.py \
        --tree_sequence_file {input.tree_sequence_file} \
        --experiment_config_filepath {params.CONFIG_FILEPATH} \
        --genome_sim_directory sampled_genome_windows \
        --window_number {wildcards.window_number} \
        --sim_number {wildcards.sim_number}
        """


# Rule: Combine metadata
rule combine_metadata:
    input:
        tree_sequence_file=rules.run_simulation.output.tree_sequence_file,
        experiment_config=CONFIG_FILEPATH,
        samples_files=lambda wildcards: [
            os.path.join(BASE_DIR, f"sampled_genome_windows/sim_{wildcards.sim_number}/window_{i}/samples.txt")
            for i in range(json.load(open(CONFIG_FILEPATH))["num_windows"])
        ]
    output:
        metadata_file=os.path.join(BASE_DIR, "sampled_genome_windows/sim_{sim_number}/metadata.txt")
    params:
        num_windows=lambda wildcards: json.load(open(CONFIG_FILEPATH))["num_windows"]
    shell:
        """
        mkdir -p $(dirname {output.metadata_file})
        for ((i=0; i<{params.num_windows}; i++)); do
            echo "{BASE_DIR}/sampled_genome_windows/sim_{wildcards.sim_number}/window_$i/window.$i.vcf.gz" >> {output.metadata_file}
        done
        """

rule calculate_LD_stats:
    input:
        pop_file_path=os.path.join(BASE_DIR, "sampled_genome_windows/sim_{sim_number}/window_{window_number}/samples.txt"),
        flat_map_file=os.path.join(BASE_DIR, "sampled_genome_windows/sim_{sim_number}/window_{window_number}/flat_map.txt"),
        metadata_file=os.path.join(BASE_DIR, "sampled_genome_windows/sim_{sim_number}/metadata.txt"),
        sampled_params_pkl=os.path.join(BASE_DIR, "simulated_parameters_and_inferences/simulation_results/sampled_params_{sim_number}.pkl")
    output:
        processed_file=os.path.join(BASE_DIR, "LD_inferences/sim_{sim_number}/window_{window_number}/ld_stats_window.{window_number}.pkl")
    params:
        window_number=lambda wildcards: wildcards.window_number
    shell:
        """
        echo "Extracting VCF file for window number {wildcards.window_number}"
        vcf_filepath=$(sed -n "$(( {wildcards.window_number} + 1 ))p" {input.metadata_file})
        echo "Processing VCF file: $vcf_filepath"
        PYTHONPATH={BASE_DIR} python {BASE_DIR}/snakemake_scripts/ld_stats.py \
            --vcf_filepath "$vcf_filepath" \
            --pop_file_path {input.pop_file_path} \
            --flat_map_path {input.flat_map_file} \
            --sim_directory {BASE_DIR}/LD_inferences \
            --sim_number {wildcards.sim_number} \
            --window_number {wildcards.window_number}
        """

rule gather_ld_stats:
    input:
        ld_stats_files=lambda wildcards: expand(
            os.path.join(BASE_DIR, "LD_inferences/sim_{sim_number}/window_{window_number}/ld_stats_window.{window_number}.pkl"),
            sim_number=wildcards.sim_number,
            window_number=range(0, experiment_config['num_windows'])
        )
    output:
        combined_ld_stats_sim=os.path.join(BASE_DIR, "combined_LD_inferences/sim_{sim_number}/combined_LD_stats_sim_{sim_number}.pkl")
    shell:
        """
        mkdir -p {BASE_DIR}/combined_LD_inferences/sim_{wildcards.sim_number}
        PYTHONPATH={BASE_DIR} python {BASE_DIR}/snakemake_scripts/combine_ld_stats.py \
            --ld_stats_files <(printf "%s\\n" {input.ld_stats_files}) \
            --sim_number {wildcards.sim_number}
        """

rule obtain_MomentsLD_feature:
    input:
        combined_ld_stats_path=rules.gather_ld_stats.output.combined_ld_stats_sim,
        sampled_params_pkl=rules.run_simulation.output.sampled_params_pkl
    output:
        software_inferences=os.path.join(BASE_DIR, "final_LD_inferences/momentsLD_inferences_sim_{sim_number}.pkl")
    shell:
        """
        mkdir -p {BASE_DIR}/final_LD_inferences
        PYTHONPATH={BASE_DIR} python {BASE_DIR}/snakemake_scripts/momentsLD_analysis.py \
            --combined_ld_stats_path {input.combined_ld_stats_path} \
            --sampled_params {input.sampled_params_pkl} \
            --experiment_config_filepath {CONFIG_FILEPATH} \
            --sim_directory {BASE_DIR}/final_LD_inferences \
            --sim_number {wildcards.sim_number}
        """

rule obtain_feature:
    input:
        sampled_params_pkl=rules.run_simulation.output.sampled_params_pkl,
        SFS=rules.run_simulation.output.sfs_file
    output:
        software_inferences=os.path.join(BASE_DIR, "moments_dadi_features/sim_{sim_number}/{analysis}/replicate_{replicate_number}/replicate_{replicate_number}.pkl")
    params:
        sim_directory=os.path.join(BASE_DIR, "moments_dadi_features"),
        analysis=lambda wildcards: "dadi" if wildcards.analysis == "dadi" else "moments"
    shell:
        """
        PYTHONPATH={BASE_DIR} python {BASE_DIR}/snakemake_scripts/obtain_single_feature.py \
            --sfs_file {input.SFS} \
            --sampled_params_pkl {input.sampled_params_pkl} \
            --experiment_config_filepath {CONFIG_FILEPATH} \
            --sim_directory {params.sim_directory} \
            --sim_number {wildcards.sim_number} \
            --replicate_number {wildcards.replicate_number}
        """

rule aggregate_top_k_results:
    input:
        dadi_files=expand(
            os.path.join(BASE_DIR, "moments_dadi_features/sim_{{sim_number}}/dadi/replicate_{replicate}/replicate_{replicate}.pkl"),
            replicate=range(json.load(open(CONFIG_FILEPATH))["k"])
        ),
        moments_files=expand(
            os.path.join(BASE_DIR, "moments_dadi_features/sim_{{sim_number}}/moments/replicate_{replicate}/replicate_{replicate}.pkl"),
            replicate=range(json.load(open(CONFIG_FILEPATH))["k"])
        ),
        sfs_file=os.path.join(BASE_DIR, "simulated_parameters_and_inferences/simulation_results/SFS_sim_{sim_number}.pkl"),
        params_file=os.path.join(BASE_DIR, "simulated_parameters_and_inferences/simulation_results/sampled_params_{sim_number}.pkl")
    output:
        software_inferences_sim=os.path.join(BASE_DIR, "moments_dadi_features/software_inferences_sim_{sim_number}.pkl")
    params:
        top_k=json.load(open(CONFIG_FILEPATH))["top_values_k"]
    shell:
        """
        PYTHONPATH={BASE_DIR} python {BASE_DIR}/snakemake_scripts/aggregate_top_k.py \
            --dadi_files {input.dadi_files} \
            --moments_files {input.moments_files} \
            --sfs_file {input.sfs_file} \
            --params_file {input.params_file} \
            --top_k {params.top_k} \
            --sim_number {wildcards.sim_number}
        """

# Rule to gather software_inferences
def gather_software_inferences(wildcards):
    try:
        return expand(
            os.path.join(BASE_DIR, "moments_dadi_features/software_inferences_sim_{sim_number}.pkl"),
            sim_number=range(0, experiment_config['num_sims_pretrain'])
        )
    except KeyError as e:
        raise KeyError(f"Key {e} not found in experiment_config. Ensure 'num_sims_pretrain' is correctly set.")
    except Exception as e:
        raise RuntimeError(f"Error gathering software inferences: {e}")


def gather_momentsLD_inferences(wildcards):
    try:
        return expand(
            os.path.join(BASE_DIR, "final_LD_inferences/momentsLD_inferences_sim_{sim_number}.pkl"),
            sim_number=range(0, experiment_config['num_sims_pretrain'])
        )
    except KeyError as e:
        raise KeyError(f"Key {e} not found in experiment_config. Ensure 'num_sims_pretrain' is correctly set.")
    except Exception as e:
        raise RuntimeError(f"Error gathering MomentsLD inferences: {e}")

rule aggregate_features:
    input:
        software_inferences=gather_software_inferences,
        momentsLD_inferences=gather_momentsLD_inferences,
        experiment_config_filepath=CONFIG_FILEPATH
    output:
        preprocessing_results=os.path.join(SIM_DIRECTORY, "preprocessing_results_obj.pkl"),
        training_features=os.path.join(SIM_DIRECTORY, "training_features.npy"),
        training_targets=os.path.join(SIM_DIRECTORY, "training_targets.npy"),
        validation_features=os.path.join(SIM_DIRECTORY, "validation_features.npy"),
        validation_targets=os.path.join(SIM_DIRECTORY, "validation_targets.npy")
    params:
        sim_directory=SIM_DIRECTORY
    shell:
        """
        mkdir -p {params.sim_directory}
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/aggregate_all_features.py \
            {input.experiment_config_filepath} \
            {params.sim_directory} \
            --software_inferences_dir {input.software_inferences:q} \
            --momentsLD_inferences_dir {input.momentsLD_inferences:q}
        """

rule postprocessing:
    input:
        experiment_config_filepath=CONFIG_FILEPATH,
        training_features_filepath=rules.aggregate_features.output.training_features,
        validation_features_filepath=rules.aggregate_features.output.validation_features,
        training_targets_filepath=rules.aggregate_features.output.training_targets,
        validation_targets_filepath=rules.aggregate_features.output.validation_targets
    params:
        SIM_DIRECTORY=SIM_DIRECTORY
    output:
        postprocessing_results=os.path.join(SIM_DIRECTORY, "postprocessing_results.pkl")
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
        postprocessing_results=rules.postprocessing.output.postprocessing_results
    output:
        features_output=os.path.join(SIM_DIRECTORY, "features_and_targets.pkl")
    params:
        sim_directory=SIM_DIRECTORY
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/extracting_features.py \
            --postprocessing_results_filepath {input.postprocessing_results} \
            --sim_directory {params.sim_directory}
        """

rule linear_evaluation:
    input:
        features_and_targets_path=rules.get_features.output.features_output,
        experiment_config_filepath=CONFIG_FILEPATH,
        color_shades_file=rules.create_experiment.output.colors_shades_file,
        main_colors_file=rules.create_experiment.output.main_colors_file
    output:
        linear_model=os.path.join(MODEL_DIRECTORY, "linear_regression_model.pkl")
    params:
        model_directory=MODEL_DIRECTORY
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/linear_evaluation.py \
            --features_and_targets_filepath {input.features_and_targets_path} \
            --experiment_config_filepath {input.experiment_config_filepath} \
            --color_shades_file {input.color_shades_file} \
            --main_colors_file {input.main_colors_file} \
            --model_directory {params.model_directory}
        """

rule train_and_predict:
    input:
        features_file=rules.get_features.output.features_output,
        color_shades_file=rules.create_experiment.output.colors_shades_file,
        main_colors_file=rules.create_experiment.output.main_colors_file
    params:
        experiment_directory=MODEL_DIRECTORY,
        MODEL_CONFIG_FILEPATH=MODEL_CONFIG_FILEPATH
    output:
        model_results=os.path.join(MODEL_DIRECTORY, "snn_results.pkl"),
        trained_model=os.path.join(MODEL_DIRECTORY, "snn_model.pth")
    shell:
        """
        PYTHONPATH={CWD} python {CWD}/snakemake_scripts/setup_trainer.py \
            --model_config {params.MODEL_CONFIG_FILEPATH} \
            --experiment_directory {params.experiment_directory} \
            --features_file {input.features_file} \
            --color_shades_file {input.color_shades_file} \
            --main_colors_file {input.main_colors_file}
        """


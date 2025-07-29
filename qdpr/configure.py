import argparse
import pydantic
from typing import Optional

def configure(config_file):
    class Settings(pydantic.BaseModel):
        # Global settings
        working_directory: str = './working_directory'
        output_filename: str = 'run_experiment.out'
        overwrite: bool = False

        # Engineering campaign settings
        campaign: str   # options: random_from_file, from_file, from_scratch
        n_steps: int = 1
        step_size: int = 32
        sequences_file: str     # path to file containing sequences for campaign == from_file
        reuse_predictions: bool = True
        predictions_file: str = './'   # path to .pkl file containing or that should contain predictions file
        prediction_type: str = 'correlation'    # options: correlation, regression
        pearsons_r_file: str = ''   # used only for prediction_type 'correlation_squared_weighted'
        selection_method: str = 'top'   # 'top' or 'probabilistic'
        compare_prediction_types: list = []
        weighted_regression: bool = False
        correlation_threshold: Optional[float] = None

        # ML model settings
        model_architecture: str = 'avgfp'
        path_to_pretrained_models: str = ''
        learning_rate: float = 0.001
        repeat_trainings: int = 1
        num_epochs: int = 100
        batch_size: int = 32
        train_split: float = 1
        early_stopping: int = 0    # if > 0 and train_split < 1, use early stopping with this patience instead of num_epochs
        normalize_training_data: bool = False  # if true, dataset labels are normalized (for training only, not for output)
        transfer_weights: str = ''
        skip_experimental_model: bool = False

        #ProSST settings
        prosst: int = 0     # if > 0, should be vocabulary size: 20, 128, 512, 1024, 2048, or 4096
        prosst_pdb_file: str = ''  # path to .pdb file to pass to ProSST (should be mutually exclusive with prosst_structure_fasta)
        prosst_sequence_fasta: str = '' # path to .fasta file to pass to ProSST
        prosst_structure_fasta: str = ''  # path to structure sequence .fasta file to pass to ProSST (should be mutually exclusive with prosst_pdb_file)
        path_to_prosst_model: str = ''

        # Protein settings
        wt_seq: str

        # Settings for initial dataset
        initial_dataset_type: str = 'file'  # options: file, random
        initial_dataset_file: str = ''      # point to file containing initial dataset
        rng_seed: int = -1

    # Execute contents of config file
    config_lines = open(config_file).readlines()
    for line in config_lines:
        exec(line)

    # Define settings namespace to store all these variables
    config_dict = {}
    config_dict.update(locals())
    settings = argparse.Namespace()
    settings.__dict__.update(Settings(**config_dict))

    return settings

"""ArC (Argument-based Consistency) computation script"""
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

from core.models.arc_config import ArCConfig
from services.arc_service import ArCService
from utils import helpers as hp
from utils.data_path_prefixes import ARC_RESULTS_PATH


def save_sample_results(results, sample_ix, model_name, data_name, explicit_prompting):
    """Save sample results to file"""
    # Remove _nontoxic suffix from data_name for saving
    base_data_name = data_name.replace('_nontoxic', '')

    if explicit_prompting == '':
        directory_path = Path(ARC_RESULTS_PATH + "_naive" + "/" + model_name.split('/')[1] + '/' + base_data_name + '/')
    else:
        directory_path = Path(ARC_RESULTS_PATH + "/" + model_name.split('/')[1] + '/' + base_data_name + '/')

    directory_path.mkdir(parents=True, exist_ok=True)
    file_path = directory_path / (str(sample_ix) + '.pkl')

    with file_path.open("wb") as f:
        pickle.dump(results, f)


def compute_arc_metrics(config: ArCConfig, specific_data_name=None):
    """
    Compute ArC metrics for all models and datasets

    Args:
        config: ArC configuration
        specific_data_name: Optional specific dataset to process
    """
    service = ArCService(config)

    # Filter data names if specific dataset requested
    data_names = [specific_data_name] if specific_data_name else service.data_names

    for data_name in data_names:
        for model_name in service.model_names:
            print(f"Processing {model_name} on {data_name} data")
            service.logger.info(f"Processing {model_name} on {data_name} data")

            # Load output tokens and parsed outputs
            output_tokens_dict = hp.get_output_tokens(model_name, data_name, config.explicit_prompting)
            parsed_output_dict = hp.get_parsed_outputs(model_name, data_name, config.explicit_prompting)

            # Determine starting index for nontoxic samples
            if '_nontoxic' in data_name:
                # Start from index 8 for nontoxic samples
                start_idx = 8
            else:
                start_idx = 0

            # Process each sample
            for sample_ix in tqdm(range(len(parsed_output_dict['justify']['input_texts']))):
                # Compute all metrics for this sample
                sample_result = service.compute_sample(
                    sample_ix, model_name, data_name,
                    output_tokens_dict, parsed_output_dict
                )

                # Save results with adjusted index for nontoxic samples
                save_idx = start_idx + sample_ix
                save_sample_results(
                    sample_result, save_idx, model_name, data_name,
                    config.explicit_prompting
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ArC (Argument-based Consistency) metrics")
    parser.add_argument(
        "--explicit_prompting", type=str, required=False, default='True',
        help="Prompt with explicit instructions (True/False)"
    )
    parser.add_argument(
        "--use_scores", type=str, required=False, default='False',
        help="Use entropy of scores instead of logits (True/False)"
    )
    parser.add_argument(
        "--similarity_model", type=str, required=False,
        default='cross-encoder/stsb-distilroberta-base',
        help="Semantic similarity model name"
    )
    parser.add_argument(
        "--data_name", type=str, required=False, default=None,
        help="Specific dataset to process (e.g., civil_comments, civil_comments_nontoxic)"
    )

    args = parser.parse_args()
    
    # Create configuration with default values
    config = ArCConfig.from_args(
        explicit_prompting=True,  # Default: use explicit prompting
        use_scores=False,         # Default: use logits-based entropy
        similarity_model=args.similarity_model
    )

    # Compute metrics
    compute_arc_metrics(config, specific_data_name=args.data_name)

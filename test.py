import pathlib
import argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import torch
import numpy as np
import soundfile as sf

from util.utils import initialize_config, load_config, mean_std


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Command line arguments.
    """
    parser = argparse.ArgumentParser(description="Script for model inference and evaluation.")
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Model configuration file (json).",
    )
    parser.add_argument(
        "--testset-config",
        required=True,
        type=pathlib.Path,
        help="Dataset configuration file (json).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        help="Path of the *.Pth file of the model.",
    )
    parser.add_argument(
        "--use-classifier",
        type=bool,
        default=True,
    )

    return parser.parse_args()


def main(args):
    """
    Main function for inference and evaluation.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    model_config = load_config(args.config)["model_config"]
    testset_config = load_config(args.testset_config)

    testset_name = args.testset_config.stem
    output_dir = args.config.parent / "results" / testset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = initialize_config(testset_config["test_dataset"])

    lightning_module = initialize_config(model_config["lightning_module"], pass_args=False).load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        config=model_config,
        sample_rate=test_dataset.sample_rate,
    )
    lightning_module = lightning_module.to(device="cuda").eval()
    test_dataloader = lightning_module.get_test_dataloader(test_dataset)

    score_records = run_inference(lightning_module, test_dataloader, args.use_classifier)
    save_results(score_records, output_dir, args.use_classifier)


def run_inference(lightning_module, dataloader, use_classifier=True):
    """
    Run inference or evaluation on a dataset using a given model.

    Args:
        lightning_module (torch.nn.Module): PyTorch model for inference or evaluation.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        mode (str):

    Returns:
        defaultdict[list]: Dictionary of score records.
    """
    score_records = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            if use_classifier:
                log_statistics = lightning_module.inference(batch)
            else:
                log_statistics = lightning_module.inference_with_noisy_phase(batch)

            for key, value in log_statistics.items():
                score_records[key].append(value)
    return score_records


def save_results(score_records, output_dir, use_classsifier):
    """
    Save inference or evaluation results to a file.

    Args:
        score_records (defaultdict[list]): Dictionary of score records.
        output_dir (pathlib.Path): Directory to save the results.
        mode (str):
    """
    log_file_path = output_dir / "log.txt"
    with log_file_path.open("a") as outfile:
        print(f"Use DNN Classifier: {use_classsifier}")
        print(f"Use DNN Classifier: {use_classsifier}", file=outfile)
        for key, value_list in score_records.items():
            mean, std = mean_std(np.asarray(value_list))
            print(f"{key}: {mean:.6f} ± {std:.6f}")
            print(f"{key}: {mean:.6f} ± {std:.6f}", file=outfile)

    df = pd.DataFrame(score_records)
    if use_classsifier:
        csv_output_path = output_dir / "use_classifier_scores.csv"
    else:
        csv_output_path = output_dir / "ldpc_decoder_scores.csv"
    df.to_csv(csv_output_path, index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)

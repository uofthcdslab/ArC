"""
Download non-toxic samples from all 5 datasets for future use.
This script downloads 8 non-toxic samples from each dataset and saves them to CSV files.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import requests
import sklearn.preprocessing
from pathlib import Path
from utils.logger_setup import setup_logger
from utils.data_path_prefixes import PROCESSED_DATA_PATH

logger = setup_logger('download_nontoxic', log_dir='logs')

def download_civil_comments_nontoxic(total_samples=8, random_state=17):
    """Download non-toxic samples from civil_comments dataset."""
    logger.info("Downloading non-toxic samples from civil_comments")

    dataset = load_dataset("civil_comments", split="train")
    df = dataset.to_pandas()

    # Filter for non-toxic (toxicity <= 0.5) and apply length constraints
    df = df[(df['toxicity'] <= 0.5) &
            (df['text'].str.len() > 64) &
            (df['text'].str.len() < 1024)]

    # Sample 8 random samples
    sampled = df.sample(n=total_samples, random_state=random_state)

    # Format output
    output = pd.DataFrame({
        'index': range(1000, 1000 + total_samples),  # Start from 1000 to avoid conflicts
        'input_text': sampled['text'].values,
        'given_label': sampled['toxicity'].values
    })

    logger.info(f"Downloaded {len(output)} non-toxic samples (max toxicity: {output['given_label'].max():.3f})")
    return output

def download_hate_explain_nontoxic(total_samples=8, random_state=17):
    """Download non-toxic samples from hate_explain dataset."""
    logger.info("Downloading non-toxic samples from hate_explain")

    # Load dataset
    data_url = "https://raw.githubusercontent.com/hate-alert/HateXplain/refs/heads/master/Data/dataset.json"
    train_ids_url = "https://raw.githubusercontent.com/hate-alert/HateXplain/refs/heads/master/Data/post_id_divisions.json"

    train_ids = requests.get(train_ids_url).json()['train']
    response_data = requests.get(data_url).json()

    texts = []
    scores = []

    for id_ in train_ids:
        annotators = response_data[id_]['annotators']
        non_normal_count = sum(a['label'] != 'normal' for a in annotators)
        score = non_normal_count / len(annotators)
        text = ' '.join(response_data[id_]['post_tokens'])

        # Filter for non-toxic (score <= 0.5) and length constraints
        if score <= 0.5 and 64 < len(text) < 1024:
            texts.append(text)
            scores.append(score)

    # Create DataFrame and sample
    df = pd.DataFrame({'input_text': texts, 'given_label': scores})
    sampled = df.sample(n=min(total_samples, len(df)), random_state=random_state)

    output = pd.DataFrame({
        'index': range(1000, 1000 + len(sampled)),
        'input_text': sampled['input_text'].values,
        'given_label': sampled['given_label'].values
    })

    logger.info(f"Downloaded {len(output)} non-toxic samples (max score: {output['given_label'].max():.3f})")
    return output

def download_real_toxicity_prompts_nontoxic(total_samples=8, random_state=17):
    """Download non-toxic samples from real_toxicity_prompts dataset."""
    logger.info("Downloading non-toxic samples from real_toxicity_prompts")

    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
    df = pd.DataFrame(dataset)

    # Extract toxicity scores and texts
    toxicity_scores = df.apply(lambda row: row['continuation']['toxicity'], axis=1)
    texts = df.apply(lambda row: f"{row['prompt']['text']}{row['continuation']['text']}", axis=1)

    # Create DataFrame
    df = pd.DataFrame({'input_text': texts, 'given_label': toxicity_scores})

    # Filter for non-toxic (toxicity <= 0.5) and length constraints
    df = df[(df['given_label'] <= 0.5) &
            (df['input_text'].str.len() > 64) &
            (df['input_text'].str.len() < 1024)]

    # Sample
    sampled = df.sample(n=min(total_samples, len(df)), random_state=random_state)

    output = pd.DataFrame({
        'index': range(1000, 1000 + len(sampled)),
        'input_text': sampled['input_text'].values,
        'given_label': sampled['given_label'].values
    })

    logger.info(f"Downloaded {len(output)} non-toxic samples (max toxicity: {output['given_label'].max():.3f})")
    return output

def download_implicit_toxicity_nontoxic(total_samples=8, random_state=17):
    """Download non-toxic samples from implicit_toxicity dataset."""
    logger.info("Downloading non-toxic samples from implicit_toxicity")

    dataset = load_dataset("jiaxin-wen/Implicit-Toxicity", data_files="train/aug-train.json")
    df = pd.DataFrame(dataset["train"])

    # Format text and filter for non-toxic (label = 0)
    df['input_text'] = df.apply(
        lambda row: f"CONTEXT:\n{row['context']}\n\nRESPONSE:\n{row['response']}",
        axis=1
    )

    df = df[(df['label'] == 0) &
            (df['input_text'].str.len() > 64) &
            (df['input_text'].str.len() < 1024)]

    # Sample
    sampled = df.sample(n=min(total_samples, len(df)), random_state=random_state)

    output = pd.DataFrame({
        'index': range(1000, 1000 + len(sampled)),
        'input_text': sampled['input_text'].values,
        'given_label': sampled['label'].values
    })

    logger.info(f"Downloaded {len(output)} non-toxic samples (all labels = 0)")
    return output

def download_toxigen_nontoxic(total_samples=8, random_state=17):
    """Download non-toxic samples from toxigen dataset."""
    logger.info("Downloading non-toxic samples from toxigen")

    dataset = load_dataset("toxigen/toxigen-data", split="train")
    df = pd.DataFrame(dataset)

    # Scale toxicity scores to [0,1]
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    toxicity_scores = scaler.fit_transform(
        np.array(df['toxicity_human']).reshape(-1, 1)
    ).flatten()

    df['given_label'] = toxicity_scores
    df = df.rename(columns={'text': 'input_text'})

    # Filter for non-toxic (scaled score <= 0.5) and length constraints
    df = df[(df['given_label'] <= 0.5) &
            (df['input_text'].str.len() > 64) &
            (df['input_text'].str.len() < 1024)]

    # Sample
    sampled = df.sample(n=min(total_samples, len(df)), random_state=random_state)

    output = pd.DataFrame({
        'index': range(1000, 1000 + len(sampled)),
        'input_text': sampled['input_text'].values,
        'given_label': sampled['given_label'].values
    })

    logger.info(f"Downloaded {len(output)} non-toxic samples (max scaled score: {output['given_label'].max():.3f})")
    return output

def main():
    """Download non-toxic samples for all 5 datasets."""
    output_dir = Path(PROCESSED_DATA_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        'civil_comments': download_civil_comments_nontoxic,
        'hate_explain': download_hate_explain_nontoxic,
        'real_toxicity_prompts': download_real_toxicity_prompts_nontoxic,
        'implicit_toxicity': download_implicit_toxicity_nontoxic,
        'toxigen': download_toxigen_nontoxic
    }

    for dataset_name, download_func in datasets.items():
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {dataset_name}")
            logger.info(f"{'='*60}")

            data = download_func()
            output_path = output_dir / f"nontoxic_{dataset_name}.csv"
            data.to_csv(output_path, index=False)

            logger.info(f"Saved to {output_path}")
            logger.info(f"Sample count: {len(data)}")
            logger.info(f"Toxicity range: [{data['given_label'].min():.3f}, {data['given_label'].max():.3f}]")

        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {str(e)}")
            continue

    logger.info(f"\n{'='*60}")
    logger.info("Download complete!")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()

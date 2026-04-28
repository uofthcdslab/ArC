# Quick Command Reference - ArC Non-Toxic Extension

## Prerequisites
```bash
conda activate haf
```

## Authentication for Gated Models

If using gated models like Llama, you need a Hugging Face token:

1. Get your token from https://huggingface.co/settings/tokens
2. Accept the model license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Add `--hf_token YOUR_TOKEN_HERE` to all `generate.py` commands below

**Alternative:** Set as environment variable to avoid repeating in each command:
```bash
export HF_TOKEN=your_token_here
```
Then use `--hf_token $HF_TOKEN` in commands.

## Step-by-Step Execution

### 1. Download Non-Toxic Samples (5 min)
```bash
python download_nontoxic_samples.py
```

### 2. Stage 1: Justify Generation and Parsing

**Generate Justify**
```bash
python generate.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments --generation_stage justify --explicit_prompting True --toxicity_type nontoxic --data_size 8 --hf_token YOUR_TOKEN_HERE --max_new_tokens 256
```

**Parse Justify**
```bash
python parse.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --stage justify --explicit_prompting True --similarity_model cross-encoder/stsb-distilroberta-base
```

### 3. Stage 2: Uphold Reasons Internal Generation and Parsing

**Generate Uphold Reasons Internal**
```bash
python generate.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --generation_stage uphold_reasons_internal --explicit_prompting True --hf_token YOUR_TOKEN_HERE --max_new_tokens 256
```

**Parse Uphold Reasons Internal**
```bash
python parse.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --stage uphold_reasons_internal --explicit_prompting True --similarity_model cross-encoder/stsb-distilroberta-base
```

### 4. Stage 3: Uphold Reasons External Generation and Parsing

**Generate Uphold Reasons External**
```bash
python generate.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --generation_stage uphold_reasons_external --explicit_prompting True --hf_token YOUR_TOKEN_HERE --max_new_tokens 256
```

**Parse Uphold Reasons External**
```bash
python parse.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --stage uphold_reasons_external --explicit_prompting True --similarity_model cross-encoder/stsb-distilroberta-base
```

### 5. Stage 4: Uphold Stance Generation and Parsing

**Generate Uphold Stance**
```bash
python generate.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --generation_stage uphold_stance --hf_token YOUR_TOKEN_HERE --max_new_tokens 256
```

**Parse Uphold Stance**
```bash
python parse.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --stage uphold_stance --similarity_model cross-encoder/stsb-distilroberta-base
```

### 6. Compute ArC Metrics (10 min)
```bash
python ArC.py --explicit_prompting True --similarity_model cross-encoder/stsb-distilroberta-base --data_name civil_comments_nontoxic
```

### 7. View Results
```bash
python view_results.py --model_name Llama-3.2-3B-Instruct --data_name civil_comments
```

## Quick Verification

**Check downloaded files:**
```bash
ls processed_sampled_input_data/nontoxic_*.csv
```

**Check generated files:**
```bash
ls llm_generated_data/Llama-3.2-3B-Instruct/civil_comments_nontoxic/
```

**Check parsed files:**
```bash
ls parsed_data/Llama-3.2-3B-Instruct/civil_comments_nontoxic/
```

**Check result count (should be 16):**
```bash
ls arc_results/Llama-3.2-3B-Instruct/civil_comments/*.pkl | wc -l
```

**Verify RS/RN split:**
```bash
python -c "import pickle; from pathlib import Path; results = [pickle.load(open(f, 'rb')) for f in sorted(Path('arc_results/Llama-3.2-3B-Instruct/civil_comments').glob('*.pkl'))]; print(f'Total: {len(results)}'); print(f'Toxic (RS): {sum(1 for r in results if \"RS\" in r)}'); print(f'Non-toxic (RN): {sum(1 for r in results if \"RN\" in r)}')"
```

## All-in-One Script

Save this as `run_nontoxic_pipeline.sh`:

```bash
#!/bin/bash
set -e

# Set your Hugging Face token here
HF_TOKEN="your_token_here"

echo "=== Step 1: Download Non-Toxic Samples ==="
python download_nontoxic_samples.py

echo "=== Step 2: Stage 1 - Justify ==="
echo "Generating..."
python generate.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments --generation_stage justify --explicit_prompting True --toxicity_type nontoxic --data_size 8 --hf_token $HF_TOKEN --max_new_tokens 256
echo "Parsing..."
python parse.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --stage justify --explicit_prompting True --similarity_model cross-encoder/stsb-distilroberta-base

echo "=== Step 3: Stage 2 - Uphold Reasons Internal ==="
echo "Generating..."
python generate.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --generation_stage uphold_reasons_internal --explicit_prompting True --hf_token $HF_TOKEN --max_new_tokens 256
echo "Parsing..."
python parse.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --stage uphold_reasons_internal --explicit_prompting True --similarity_model cross-encoder/stsb-distilroberta-base

echo "=== Step 4: Stage 3 - Uphold Reasons External ==="
echo "Generating..."
python generate.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --generation_stage uphold_reasons_external --explicit_prompting True --hf_token $HF_TOKEN --max_new_tokens 256
echo "Parsing..."
python parse.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --stage uphold_reasons_external --explicit_prompting True --similarity_model cross-encoder/stsb-distilroberta-base

echo "=== Step 5: Stage 4 - Uphold Stance ==="
echo "Generating..."
python generate.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --generation_stage uphold_stance --hf_token $HF_TOKEN --max_new_tokens 256
echo "Parsing..."
python parse.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_name civil_comments_nontoxic --stage uphold_stance --similarity_model cross-encoder/stsb-distilroberta-base

echo "=== Step 6: Compute ArC Metrics ==="
python ArC.py --explicit_prompting True --similarity_model cross-encoder/stsb-distilroberta-base --data_name civil_comments_nontoxic

echo "=== Step 7: View Results ==="
python view_results.py --model_name Llama-3.2-3B-Instruct --data_name civil_comments

echo "=== Pipeline Complete! ==="
```

Make executable and run:
```bash
chmod +x run_nontoxic_pipeline.sh
./run_nontoxic_pipeline.sh
```

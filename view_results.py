import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_arc_results(model_name, data_name, results_path="arc_results"):
    """Load all ArC results for a specific model and dataset"""
    
    # Get the model short name (e.g., "Llama-3.1-8B-Instruct" from "meta-llama/Llama-3.1-8B-Instruct")
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    
    directory_path = Path(results_path) / model_short / data_name
    
    if not directory_path.exists():
        print(f"ERROR: Directory not found: {directory_path}")
        return None
    
    # Load all pickle files
    results = []
    pkl_files = sorted(directory_path.glob("*.pkl"))
    
    if not pkl_files:
        print(f"ERROR: No result files found in {directory_path}")
        return None
    
    print(f"Found {len(pkl_files)} result files")
    
    for pkl_file in pkl_files:
        sample_idx = int(pkl_file.stem)
        with open(pkl_file, 'rb') as f:
            sample_result = pickle.load(f)
            sample_result['sample_idx'] = sample_idx
            results.append(sample_result)
    
    return results

def summarize_arc_metrics(results):
    """Summarize ArC metrics across all samples"""
    
    if not results:
        print("ERROR: No results to summarize")
        return None
    
    summary = {}
    
    # Collect metrics
    metrics_data = {
        'initial_decision_confidence': [],
        'internal_decision_confidence': [],
        'external_decision_confidence': [],
        'DiS_avg': []
    }
    
    # SoS, UII, UEI, RS, RN metrics (per-reason)
    sos_values = []
    uii_values = []
    uei_values = []
    rs_values = []
    rn_values = []
    
    for result in results:
        # Single-value metrics
        for key in metrics_data.keys():
            if key in result and not pd.isna(result[key]):
                metrics_data[key].append(result[key])
        
        # SoS (Sufficiency of Stance)
        if 'SoS' in result:
            for reason_key, value in result['SoS'].items():
                if not pd.isna(value):
                    sos_values.append(value)
        
        # UII (Uncertainty in Internal Informativeness)
        if 'UII' in result:
            for reason_key, value in result['UII'].items():
                if not pd.isna(value):
                    uii_values.append(value)
        
        # UEI (Uncertainty in External Informativeness)
        if 'UEI' in result:
            for reason_key, value in result['UEI'].items():
                if not pd.isna(value):
                    uei_values.append(value)
        
        # RS (Reason Sufficiency)
        if 'RS' in result:
            for subsample_idx, value in result['RS'].items():
                if not pd.isna(value):
                    rs_values.append(value)
        
        # RN (Reason Necessity)
        if 'RN' in result:
            for subsample_idx, value in result['RN'].items():
                if not pd.isna(value):
                    rn_values.append(value)
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("ArC METRICS SUMMARY")
    print("="*80)
    
    print("\nRELEVANCE DIMENSION:")
    print("-" * 80)
    if sos_values:
        print(f"  SoS (Sufficiency of Stance):")
        print(f"    Mean: {np.mean(sos_values):.4f} | Std: {np.std(sos_values):.4f}")
        print(f"    Min: {np.min(sos_values):.4f} | Max: {np.max(sos_values):.4f}")
    
    if metrics_data['DiS_avg']:
        print(f"\n  DiS-Avg (Diversity of Stance - Average):")
        print(f"    Mean: {np.mean(metrics_data['DiS_avg']):.4f} | Std: {np.std(metrics_data['DiS_avg']):.4f}")
    
    print("\nINTERNAL RELIANCE DIMENSION:")
    print("-" * 80)
    if uii_values:
        print(f"  UII (Uncertainty in Internal Informativeness):")
        print(f"    Mean: {np.mean(uii_values):.4f} | Std: {np.std(uii_values):.4f}")
        print(f"    Min: {np.min(uii_values):.4f} | Max: {np.max(uii_values):.4f}")
    
    print("\nEXTERNAL RELIANCE DIMENSION:")
    print("-" * 80)
    if uei_values:
        print(f"  UEI (Uncertainty in External Informativeness):")
        print(f"    Mean: {np.mean(uei_values):.4f} | Std: {np.std(uei_values):.4f}")
        print(f"    Min: {np.min(uei_values):.4f} | Max: {np.max(uei_values):.4f}")
    
    print("\nINDIVIDUAL RELIANCE DIMENSION:")
    print("-" * 80)
    if rs_values:
        print(f"  RS (Reason Sufficiency):")
        print(f"    Mean: {np.mean(rs_values):.4f} | Std: {np.std(rs_values):.4f}")
        print(f"    Min: {np.min(rs_values):.4f} | Max: {np.max(rs_values):.4f}")
    
    if rn_values:
        print(f"\n  RN (Reason Necessity):")
        print(f"    Mean: {np.mean(rn_values):.4f} | Std: {np.std(rn_values):.4f}")
        print(f"    Min: {np.min(rn_values):.4f} | Max: {np.max(rn_values):.4f}")
    
    print("\nDECISION CONFIDENCE:")
    print("-" * 80)
    for conf_type in ['initial', 'internal', 'external']:
        key = f'{conf_type}_decision_confidence'
        if metrics_data[key]:
            print(f"  {conf_type.capitalize()}: Mean={np.mean(metrics_data[key]):.4f}, Std={np.std(metrics_data[key]):.4f}")
    
    print("\n" + "="*80)
    print(f"Total samples analyzed: {len(results)}")
    print("="*80 + "\n")
    
    return {
        'SoS': sos_values,
        'UII': uii_values,
        'UEI': uei_values,
        'RS': rs_values,
        'RN': rn_values,
        'metrics_data': metrics_data
    }

def view_sample_detail(results, sample_idx):
    """View detailed results for a specific sample"""
    
    sample = next((r for r in results if r['sample_idx'] == sample_idx), None)
    
    if not sample:
        print(f"ERROR: Sample {sample_idx} not found")
        return
    
    print(f"\n{'='*80}")
    print(f"DETAILED RESULTS FOR SAMPLE {sample_idx}")
    print(f"{'='*80}\n")
    
    for key, value in sample.items():
        if key == 'sample_idx':
            continue
        print(f"{key}:")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"  {value}")
        print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View ArC computation results")
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", 
                       help="Model name (short form like 'Llama-3.1-8B-Instruct')")
    parser.add_argument("--data_name", type=str, default="civil_comments",
                       help="Dataset name")
    parser.add_argument("--sample_idx", type=int, default=None,
                       help="View specific sample details")
    parser.add_argument("--results_path", type=str, default="arc_results",
                       help="Path to results directory")
    
    args = parser.parse_args()
    
    # Load results
    print(f"\nLoading results for {args.model_name} on {args.data_name}...")
    results = load_arc_results(args.model_name, args.data_name, args.results_path)
    
    if results:
        if args.sample_idx is not None:
            # View specific sample
            view_sample_detail(results, args.sample_idx)
        else:
            # View summary
            summary = summarize_arc_metrics(results)
    else:
        print("\nAvailable models and datasets:")
        results_path = Path(args.results_path)
        if results_path.exists():
            for model_dir in results_path.iterdir():
                if model_dir.is_dir():
                    print(f"\n  Model: {model_dir.name}")
                    for data_dir in model_dir.iterdir():
                        if data_dir.is_dir():
                            pkl_count = len(list(data_dir.glob("*.pkl")))
                            print(f"    - {data_dir.name} ({pkl_count} samples)")

"""Results retrieval endpoints"""
from fastapi import APIRouter, HTTPException, Query
from typing import List
import pickle
import numpy as np
from pathlib import Path
from api.schemas.responses import SummaryResponse, SampleResultResponse, MetricSummary

router = APIRouter()


def load_sample_result(model_name: str, data_name: str, sample_idx: int):
    """Load a single sample result"""
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    file_path = Path("arc_results") / model_short / data_name / f"{sample_idx}.pkl"
    
    if not file_path.exists():
        return None
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_all_results(model_name: str, data_name: str):
    """Load all results for a model/dataset"""
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    directory_path = Path("arc_results") / model_short / data_name
    
    if not directory_path.exists():
        return None
    
    results = []
    pkl_files = sorted(directory_path.glob("*.pkl"))
    
    for pkl_file in pkl_files:
        sample_idx = int(pkl_file.stem)
        with open(pkl_file, 'rb') as f:
            sample_result = pickle.load(f)
            sample_result['sample_idx'] = sample_idx
            results.append(sample_result)
    
    return results


def calculate_metric_summary(values: List[float]) -> MetricSummary:
    """Calculate summary statistics for a metric"""
    clean_values = [v for v in values if v is not None and not np.isnan(v)]
    
    if not clean_values:
        return MetricSummary(mean=0, std=0, min=0, max=0, count=0)
    
    return MetricSummary(
        mean=float(np.mean(clean_values)),
        std=float(np.std(clean_values)),
        min=float(np.min(clean_values)),
        max=float(np.max(clean_values)),
        count=len(clean_values)
    )


@router.get("/sample/{model_name}/{data_name}/{sample_idx}", response_model=SampleResultResponse)
async def get_sample_result(model_name: str, data_name: str, sample_idx: int):
    """Get ArC results for a specific sample"""
    result = load_sample_result(model_name, data_name, sample_idx)
    
    if result is None:
        raise HTTPException(status_code=404, detail=f"Sample {sample_idx} not found")
    
    return SampleResultResponse(
        sample_idx=sample_idx,
        model_name=model_name,
        data_name=data_name,
        result=result
    )


@router.get("/summary/{model_name}/{data_name}", response_model=SummaryResponse)
async def get_summary(model_name: str, data_name: str):
    """Get aggregated statistics for model/dataset"""
    results = load_all_results(model_name, data_name)
    
    if results is None:
        raise HTTPException(status_code=404, detail=f"No results found for {model_name} on {data_name}")
    
    # Collect metric values
    metrics = {}
    
    # Single-value metrics
    for metric_name in ['initial_decision_confidence', 'internal_decision_confidence', 
                       'external_decision_confidence', 'internal_del_pe', 'external_del_pe',
                       'DiS_dpp', 'DiS_avg']:
        values = [r.get(metric_name) for r in results if metric_name in r]
        if values:
            metrics[metric_name] = calculate_metric_summary(values)
    
    # Per-reason metrics (SoS, UII, UEI)
    for metric_name in ['SoS', 'UII', 'UEI']:
        values = []
        for result in results:
            if metric_name in result:
                for reason_key, value in result[metric_name].items():
                    if not np.isnan(value):
                        values.append(value)
        if values:
            metrics[metric_name] = calculate_metric_summary(values)
    
    # Per-subsample metrics (RS, RN)
    for metric_name in ['RS', 'RN']:
        values = []
        for result in results:
            if metric_name in result:
                for subsample_idx, value in result[metric_name].items():
                    if not np.isnan(value):
                        values.append(value)
        if values:
            metrics[metric_name] = calculate_metric_summary(values)
    
    return SummaryResponse(
        model_name=model_name,
        data_name=data_name,
        total_samples=len(results),
        metrics=metrics
    )


@router.get("/compare")
async def compare_models(
    model_names: List[str] = Query(..., description="List of model names to compare"),
    data_name: str = Query(..., description="Dataset name")
):
    """Compare multiple models on the same dataset"""
    comparison = {}
    
    for model_name in model_names:
        try:
            results = load_all_results(model_name, data_name)
            if results is None:
                comparison[model_name] = {"error": "No results found"}
                continue
            
            # Calculate metrics for this model
            model_metrics = {}
            
            # Per-reason metrics
            for metric_name in ['SoS', 'UII', 'UEI']:
                values = []
                for result in results:
                    if metric_name in result:
                        for reason_key, value in result[metric_name].items():
                            if not np.isnan(value):
                                values.append(value)
                if values:
                    model_metrics[metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values))
                    }
            
            # Per-subsample metrics
            for metric_name in ['RS', 'RN']:
                values = []
                for result in results:
                    if metric_name in result:
                        for subsample_idx, value in result[metric_name].items():
                            if not np.isnan(value):
                                values.append(value)
                if values:
                    model_metrics[metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values))
                    }
            
            comparison[model_name] = model_metrics
            
        except Exception as e:
            comparison[model_name] = {"error": str(e)}
    
    return comparison


@router.get("/list")
async def list_available_results():
    """List all available model/dataset combinations"""
    results_path = Path("arc_results")
    available = []
    
    if results_path.exists():
        for model_dir in results_path.iterdir():
            if model_dir.is_dir():
                for data_dir in model_dir.iterdir():
                    if data_dir.is_dir():
                        sample_count = len(list(data_dir.glob("*.pkl")))
                        available.append({
                            "model": model_dir.name,
                            "dataset": data_dir.name,
                            "samples": sample_count
                        })
    
    return available

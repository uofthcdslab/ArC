"""Models and datasets endpoints"""
from fastapi import APIRouter, HTTPException
from typing import List
import json
from pathlib import Path
from api.schemas.responses import ModelInfo, DatasetInfo

router = APIRouter()


@router.get("/list", response_model=List[str])
async def list_models():
    """List all available models"""
    try:
        with open("utils/model_size_map.json", "r") as file:
            model_size = json.load(file)
        return list(model_size.keys())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")


@router.get("/datasets", response_model=List[str])
async def list_datasets():
    """List all available datasets"""
    try:
        with open("utils/data_path_map.json", "r") as file:
            data_path = json.load(file)
        return list(data_path.keys())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading datasets: {str(e)}")


@router.get("/{model_name}/info", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    try:
        # Get model short name
        model_short = model_name.split('/')[-1] if '/' in model_name else model_name
        
        # Check available datasets for this model
        results_path = Path("arc_results") / model_short
        available_datasets = []
        
        if results_path.exists():
            for data_dir in results_path.iterdir():
                if data_dir.is_dir():
                    available_datasets.append(data_dir.name)
        
        return ModelInfo(
            name=model_name,
            short_name=model_short,
            available_datasets=available_datasets
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@router.get("/dataset/{data_name}/info", response_model=DatasetInfo)
async def get_dataset_info(data_name: str):
    """Get information about a specific dataset"""
    try:
        results_path = Path("arc_results")
        available_models = []
        sample_count = None
        
        if results_path.exists():
            for model_dir in results_path.iterdir():
                if model_dir.is_dir():
                    data_dir = model_dir / data_name
                    if data_dir.exists():
                        available_models.append(model_dir.name)
                        if sample_count is None:
                            sample_count = len(list(data_dir.glob("*.pkl")))
        
        return DatasetInfo(
            name=data_name,
            available_models=available_models,
            sample_count=sample_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset info: {str(e)}")

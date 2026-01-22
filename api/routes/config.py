"""Configuration management endpoints"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
from pathlib import Path

router = APIRouter()


class ConfigUpdate(BaseModel):
    """Configuration update request"""
    config_data: Dict[str, Any]


class ConfigItemCreate(BaseModel):
    """Create a new configuration item"""
    key: str
    value: Any


class ConfigItemUpdate(BaseModel):
    """Update a configuration item"""
    value: Any


# ===== MODELS CONFIGURATION =====

@router.get("/models")
async def get_model_config():
    """Read all model configurations"""
    try:
        with open("utils/model_size_map.json", "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model config: {str(e)}")


@router.get("/models/{model_name}")
async def get_model_item(model_name: str):
    """Read a specific model configuration"""
    try:
        with open("utils/model_size_map.json", "r") as f:
            config = json.load(f)
        
        if model_name not in config:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        return {model_name: config[model_name]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model config: {str(e)}")


@router.post("/models")
async def create_model_item(item: ConfigItemCreate):
    """Create a new model configuration"""
    try:
        with open("utils/model_size_map.json", "r") as f:
            config = json.load(f)
        
        if item.key in config:
            raise HTTPException(status_code=400, detail=f"Model '{item.key}' already exists")
        
        config[item.key] = item.value
        
        with open("utils/model_size_map.json", "w") as f:
            json.dump(config, f, indent=4)
        
        return {"success": True, "message": f"Model '{item.key}' created"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating model: {str(e)}")


@router.put("/models")
async def update_all_models(config: ConfigUpdate):
    """Update entire model configuration"""
    try:
        with open("utils/model_size_map.json", "w") as f:
            json.dump(config.config_data, f, indent=4)
        return {"success": True, "message": "Model configuration updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating model config: {str(e)}")


@router.put("/models/{model_name}")
async def update_model_item(model_name: str, item: ConfigItemUpdate):
    """Update a specific model configuration"""
    try:
        with open("utils/model_size_map.json", "r") as f:
            config = json.load(f)
        
        if model_name not in config:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        config[model_name] = item.value
        
        with open("utils/model_size_map.json", "w") as f:
            json.dump(config, f, indent=4)
        
        return {"success": True, "message": f"Model '{model_name}' updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating model: {str(e)}")


@router.delete("/models/{model_name}")
async def delete_model_item(model_name: str):
    """Delete a specific model configuration"""
    try:
        with open("utils/model_size_map.json", "r") as f:
            config = json.load(f)
        
        if model_name not in config:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        del config[model_name]
        
        with open("utils/model_size_map.json", "w") as f:
            json.dump(config, f, indent=4)
        
        return {"success": True, "message": f"Model '{model_name}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")


# ===== DATASETS CONFIGURATION =====

@router.get("/datasets")
async def get_dataset_config():
    """Read all dataset configurations"""
    try:
        with open("utils/data_path_map.json", "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset config: {str(e)}")


@router.get("/datasets/{dataset_name}")
async def get_dataset_item(dataset_name: str):
    """Read a specific dataset configuration"""
    try:
        with open("utils/data_path_map.json", "r") as f:
            config = json.load(f)
        
        if dataset_name not in config:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        return {dataset_name: config[dataset_name]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset config: {str(e)}")


@router.post("/datasets")
async def create_dataset_item(item: ConfigItemCreate):
    """Create a new dataset configuration"""
    try:
        with open("utils/data_path_map.json", "r") as f:
            config = json.load(f)
        
        if item.key in config:
            raise HTTPException(status_code=400, detail=f"Dataset '{item.key}' already exists")
        
        config[item.key] = item.value
        
        with open("utils/data_path_map.json", "w") as f:
            json.dump(config, f, indent=4)
        
        return {"success": True, "message": f"Dataset '{item.key}' created"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating dataset: {str(e)}")


@router.put("/datasets")
async def update_all_datasets(config: ConfigUpdate):
    """Update entire dataset configuration"""
    try:
        with open("utils/data_path_map.json", "w") as f:
            json.dump(config.config_data, f, indent=4)
        return {"success": True, "message": "Dataset configuration updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating dataset config: {str(e)}")


@router.put("/datasets/{dataset_name}")
async def update_dataset_item(dataset_name: str, item: ConfigItemUpdate):
    """Update a specific dataset configuration"""
    try:
        with open("utils/data_path_map.json", "r") as f:
            config = json.load(f)
        
        if dataset_name not in config:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        config[dataset_name] = item.value
        
        with open("utils/data_path_map.json", "w") as f:
            json.dump(config, f, indent=4)
        
        return {"success": True, "message": f"Dataset '{dataset_name}' updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating dataset: {str(e)}")


@router.delete("/datasets/{dataset_name}")
async def delete_dataset_item(dataset_name: str):
    """Delete a specific dataset configuration"""
    try:
        with open("utils/data_path_map.json", "r") as f:
            config = json.load(f)
        
        if dataset_name not in config:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        del config[dataset_name]
        
        with open("utils/data_path_map.json", "w") as f:
            json.dump(config, f, indent=4)
        
        return {"success": True, "message": f"Dataset '{dataset_name}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}")


# ===== HYPERPARAMETERS CONFIGURATION =====

@router.get("/hyperparams")
async def get_hyperparams():
    """Read all ArC hyperparameters"""
    try:
        with open("utils/arc_hyperparams.py", "r") as f:
            content = f.read()
        
        hyperparams = {}
        for line in content.split('\n'):
            if '=' in line and not line.strip().startswith('#'):
                parts = line.split('=')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    try:
                        hyperparams[key] = float(value)
                    except:
                        pass
        
        return hyperparams
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading hyperparams: {str(e)}")


@router.get("/hyperparams/{param_name}")
async def get_hyperparam_item(param_name: str):
    """Read a specific hyperparameter"""
    try:
        hyperparams = await get_hyperparams()
        
        if param_name not in hyperparams:
            raise HTTPException(status_code=404, detail=f"Hyperparameter '{param_name}' not found")
        
        return {param_name: hyperparams[param_name]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading hyperparam: {str(e)}")


@router.post("/hyperparams")
async def create_hyperparam_item(item: ConfigItemCreate):
    """Create a new hyperparameter"""
    try:
        hyperparams = await get_hyperparams()
        
        if item.key in hyperparams:
            raise HTTPException(status_code=400, detail=f"Hyperparameter '{item.key}' already exists")
        
        hyperparams[item.key] = item.value
        
        lines = [f"{key} = {value}" for key, value in hyperparams.items()]
        with open("utils/arc_hyperparams.py", "w") as f:
            f.write('\n'.join(lines) + '\n')
        
        return {"success": True, "message": f"Hyperparameter '{item.key}' created"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating hyperparam: {str(e)}")


@router.put("/hyperparams")
async def update_all_hyperparams(config: ConfigUpdate):
    """Update all hyperparameters"""
    try:
        lines = [f"{key} = {value}" for key, value in config.config_data.items()]
        with open("utils/arc_hyperparams.py", "w") as f:
            f.write('\n'.join(lines) + '\n')
        
        return {"success": True, "message": "Hyperparameters updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating hyperparams: {str(e)}")


@router.put("/hyperparams/{param_name}")
async def update_hyperparam_item(param_name: str, item: ConfigItemUpdate):
    """Update a specific hyperparameter"""
    try:
        hyperparams = await get_hyperparams()
        
        if param_name not in hyperparams:
            raise HTTPException(status_code=404, detail=f"Hyperparameter '{param_name}' not found")
        
        hyperparams[param_name] = item.value
        
        lines = [f"{key} = {value}" for key, value in hyperparams.items()]
        with open("utils/arc_hyperparams.py", "w") as f:
            f.write('\n'.join(lines) + '\n')
        
        return {"success": True, "message": f"Hyperparameter '{param_name}' updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating hyperparam: {str(e)}")


@router.delete("/hyperparams/{param_name}")
async def delete_hyperparam_item(param_name: str):
    """Delete a specific hyperparameter"""
    try:
        hyperparams = await get_hyperparams()
        
        if param_name not in hyperparams:
            raise HTTPException(status_code=404, detail=f"Hyperparameter '{param_name}' not found")
        
        del hyperparams[param_name]
        
        lines = [f"{key} = {value}" for key, value in hyperparams.items()]
        with open("utils/arc_hyperparams.py", "w") as f:
            f.write('\n'.join(lines) + '\n')
        
        return {"success": True, "message": f"Hyperparameter '{param_name}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting hyperparam: {str(e)}")


# ===== PROMPTS CONFIGURATION =====

@router.get("/prompts")
async def get_prompts():
    """Read all prompt instructions"""
    try:
        with open("utils/prompt_instructions.json", "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading prompts: {str(e)}")


@router.get("/prompts/{prompt_key}")
async def get_prompt_item(prompt_key: str):
    """Read a specific prompt instruction"""
    try:
        with open("utils/prompt_instructions.json", "r") as f:
            config = json.load(f)
        
        if prompt_key not in config:
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_key}' not found")
        
        return {prompt_key: config[prompt_key]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading prompt: {str(e)}")


@router.post("/prompts")
async def create_prompt_item(item: ConfigItemCreate):
    """Create a new prompt instruction"""
    try:
        with open("utils/prompt_instructions.json", "r") as f:
            config = json.load(f)
        
        if item.key in config:
            raise HTTPException(status_code=400, detail=f"Prompt '{item.key}' already exists")
        
        config[item.key] = item.value
        
        with open("utils/prompt_instructions.json", "w") as f:
            json.dump(config, f, indent=4)
        
        return {"success": True, "message": f"Prompt '{item.key}' created"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating prompt: {str(e)}")


@router.put("/prompts")
async def update_all_prompts(config: ConfigUpdate):
    """Update all prompt instructions"""
    try:
        with open("utils/prompt_instructions.json", "w") as f:
            json.dump(config.config_data, f, indent=4)
        return {"success": True, "message": "Prompt instructions updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating prompts: {str(e)}")


@router.put("/prompts/{prompt_key}")
async def update_prompt_item(prompt_key: str, item: ConfigItemUpdate):
    """Update a specific prompt instruction"""
    try:
        with open("utils/prompt_instructions.json", "r") as f:
            config = json.load(f)
        
        if prompt_key not in config:
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_key}' not found")
        
        config[prompt_key] = item.value
        
        with open("utils/prompt_instructions.json", "w") as f:
            json.dump(config, f, indent=4)
        
        return {"success": True, "message": f"Prompt '{prompt_key}' updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating prompt: {str(e)}")


@router.delete("/prompts/{prompt_key}")
async def delete_prompt_item(prompt_key: str):
    """Delete a specific prompt instruction"""
    try:
        with open("utils/prompt_instructions.json", "r") as f:
            config = json.load(f)
        
        if prompt_key not in config:
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_key}' not found")
        
        del config[prompt_key]
        
        with open("utils/prompt_instructions.json", "w") as f:
            json.dump(config, f, indent=4)
        
        return {"success": True, "message": f"Prompt '{prompt_key}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting prompt: {str(e)}")


# ===== PATH PREFIXES CONFIGURATION =====

@router.get("/paths")
async def get_path_prefixes():
    """Read all data path prefixes"""
    try:
        with open("utils/data_path_prefixes.py", "r") as f:
            content = f.read()
        
        paths = {}
        for line in content.split('\n'):
            if '=' in line and not line.strip().startswith('#'):
                parts = line.split('=')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().strip("'\"")
                    paths[key] = value
        
        return paths
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading path prefixes: {str(e)}")


@router.get("/paths/{path_key}")
async def get_path_item(path_key: str):
    """Read a specific path prefix"""
    try:
        paths = await get_path_prefixes()
        
        if path_key not in paths:
            raise HTTPException(status_code=404, detail=f"Path '{path_key}' not found")
        
        return {path_key: paths[path_key]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading path: {str(e)}")


@router.post("/paths")
async def create_path_item(item: ConfigItemCreate):
    """Create a new path prefix"""
    try:
        paths = await get_path_prefixes()
        
        if item.key in paths:
            raise HTTPException(status_code=400, detail=f"Path '{item.key}' already exists")
        
        paths[item.key] = item.value
        
        lines = [f"{key} = '{value}'" for key, value in paths.items()]
        with open("utils/data_path_prefixes.py", "w") as f:
            f.write('\n'.join(lines) + '\n')
        
        return {"success": True, "message": f"Path '{item.key}' created"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating path: {str(e)}")


@router.put("/paths")
async def update_all_paths(config: ConfigUpdate):
    """Update all path prefixes"""
    try:
        lines = [f"{key} = '{value}'" for key, value in config.config_data.items()]
        with open("utils/data_path_prefixes.py", "w") as f:
            f.write('\n'.join(lines) + '\n')
        
        return {"success": True, "message": "Path prefixes updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating path prefixes: {str(e)}")


@router.put("/paths/{path_key}")
async def update_path_item(path_key: str, item: ConfigItemUpdate):
    """Update a specific path prefix"""
    try:
        paths = await get_path_prefixes()
        
        if path_key not in paths:
            raise HTTPException(status_code=404, detail=f"Path '{path_key}' not found")
        
        paths[path_key] = item.value
        
        lines = [f"{key} = '{value}'" for key, value in paths.items()]
        with open("utils/data_path_prefixes.py", "w") as f:
            f.write('\n'.join(lines) + '\n')
        
        return {"success": True, "message": f"Path '{path_key}' updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating path: {str(e)}")


@router.delete("/paths/{path_key}")
async def delete_path_item(path_key: str):
    """Delete a specific path prefix"""
    try:
        paths = await get_path_prefixes()
        
        if path_key not in paths:
            raise HTTPException(status_code=404, detail=f"Path '{path_key}' not found")
        
        del paths[path_key]
        
        lines = [f"{key} = '{value}'" for key, value in paths.items()]
        with open("utils/data_path_prefixes.py", "w") as f:
            f.write('\n'.join(lines) + '\n')
        
        return {"success": True, "message": f"Path '{path_key}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting path: {str(e)}")


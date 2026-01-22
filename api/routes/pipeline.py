"""Pipeline execution endpoints"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import subprocess
import json

router = APIRouter()


class GenerateRequest(BaseModel):
    """Request to generate LLM outputs"""
    data_name: str
    model_name: str
    generation_stage: str  # initial, internal, external, individual
    explicit_prompting: bool = True
    batch_size: int = 16
    data_size: Optional[int] = None


class ParseRequest(BaseModel):
    """Request to parse LLM outputs"""
    data_name: str
    model_name: str
    stage: str  # initial, internal, external, individual


class PipelineRequest(BaseModel):
    """Request to run full pipeline"""
    data_name: str
    model_name: str
    explicit_prompting: bool = True
    batch_size: int = 16
    data_size: Optional[int] = None
    stages: List[str] = ["initial", "internal", "external", "individual"]


def run_generate(request: GenerateRequest):
    """Run generate.py script"""
    cmd = [
        "python", "generate.py",
        "--data_name", request.data_name,
        "--model_name", request.model_name,
        "--generation_stage", request.generation_stage,
        "--explicit_prompting", str(request.explicit_prompting),
        "--batch_size", str(request.batch_size)
    ]
    
    if request.data_size:
        cmd.extend(["--data_size", str(request.data_size)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def run_parse(request: ParseRequest):
    """Run parse.py script"""
    cmd = [
        "python", "parse.py",
        "--data_name", request.data_name,
        "--model_name", request.model_name,
        "--stage", request.stage
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


@router.post("/generate")
async def generate_outputs(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate LLM outputs for a specific stage"""
    try:
        background_tasks.add_task(run_generate, request)
        return {
            "success": True,
            "message": f"Generation started for {request.model_name} on {request.data_name} ({request.generation_stage} stage)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/parse")
async def parse_outputs(request: ParseRequest, background_tasks: BackgroundTasks):
    """Parse LLM outputs for a specific stage"""
    try:
        background_tasks.add_task(run_parse, request)
        return {
            "success": True,
            "message": f"Parsing started for {request.model_name} on {request.data_name} ({request.stage} stage)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@router.post("/full")
async def run_full_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Run the full pipeline: generate → parse → compute for all stages
    This runs in the background and can take a long time
    """
    try:
        def execute_pipeline():
            results = []
            
            # Run each stage
            for stage in request.stages:
                # Generate
                gen_req = GenerateRequest(
                    data_name=request.data_name,
                    model_name=request.model_name,
                    generation_stage=stage,
                    explicit_prompting=request.explicit_prompting,
                    batch_size=request.batch_size,
                    data_size=request.data_size
                )
                gen_result = run_generate(gen_req)
                results.append({"stage": stage, "step": "generate", "result": gen_result})
                
                if not gen_result["success"]:
                    break
                
                # Parse
                parse_req = ParseRequest(
                    data_name=request.data_name,
                    model_name=request.model_name,
                    stage=stage
                )
                parse_result = run_parse(parse_req)
                results.append({"stage": stage, "step": "parse", "result": parse_result})
                
                if not parse_result["success"]:
                    break
            
            # Compute ArC metrics
            from core.models.arc_config import ArCConfig
            from services.arc_service import ArCService
            from utils import helpers as hp
            from pathlib import Path
            import pickle
            
            config = ArCConfig(
                explicit_prompting='_explicit' if request.explicit_prompting else '',
                use_scores=False,
                similarity_model="cross-encoder/stsb-distilroberta-base"
            )
            
            service = ArCService(config)
            output_tokens_dict = hp.get_output_tokens(request.model_name, request.data_name, config.explicit_prompting)
            parsed_output_dict = hp.get_parsed_outputs(request.model_name, request.data_name, config.explicit_prompting)
            
            for sample_ix in range(len(parsed_output_dict['initial']['input_texts'])):
                sample_result = service.compute_sample(
                    sample_ix, request.model_name, request.data_name,
                    output_tokens_dict, parsed_output_dict
                )
                
                # Save results
                from utils.data_path_prefixes import ARC_RESULTS_PATH
                model_short = request.model_name.split('/')[-1]
                directory_path = Path(ARC_RESULTS_PATH) / model_short / request.data_name
                directory_path.mkdir(parents=True, exist_ok=True)
                
                with open(directory_path / f"{sample_ix}.pkl", "wb") as f:
                    pickle.dump(sample_result, f)
            
            results.append({"stage": "compute", "step": "arc", "result": {"success": True}})
            
            return results
        
        background_tasks.add_task(execute_pipeline)
        
        return {
            "success": True,
            "message": f"Full pipeline started for {request.model_name} on {request.data_name}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

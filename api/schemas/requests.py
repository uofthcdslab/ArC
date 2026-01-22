"""Request schemas for ArC API"""
from pydantic import BaseModel, Field
from typing import List, Optional


class ArCConfigSchema(BaseModel):
    """ArC configuration schema"""
    explicit_prompting: bool = Field(default=True, description="Use explicit prompting")
    use_scores: bool = Field(default=False, description="Use scores instead of logits")
    similarity_model: str = Field(
        default="cross-encoder/stsb-distilroberta-base",
        description="Similarity model name"
    )


class ComputeRequest(BaseModel):
    """Request to compute ArC metrics for a model/dataset"""
    model_name: str = Field(..., description="Model name")
    data_name: str = Field(..., description="Dataset name")
    config: ArCConfigSchema = Field(default_factory=ArCConfigSchema)


class BatchComputeRequest(BaseModel):
    """Request to compute ArC metrics for multiple samples"""
    model_name: str = Field(..., description="Model name")
    data_name: str = Field(..., description="Dataset name")
    sample_indices: Optional[List[int]] = Field(None, description="Specific sample indices to compute")
    config: ArCConfigSchema = Field(default_factory=ArCConfigSchema)

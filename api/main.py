"""FastAPI main application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import compute, results, models, config, pipeline

app = FastAPI(
    title="ArC API",
    description="Argument-based Consistency Metrics API",
    version="1.0.0"
)

# CORS for Gradio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(compute.router, prefix="/api/v1/compute", tags=["compute"])
app.include_router(results.router, prefix="/api/v1/results", tags=["results"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(config.router, prefix="/api/v1/config", tags=["config"])
app.include_router(pipeline.router, prefix="/api/v1/pipeline", tags=["pipeline"])

@app.get("/")
async def root():
    return {"message": "ArC API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

"""Gradio UI for ArC Dashboard"""
import gradio as gr
import requests
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=2)
        if response.status_code == 200:
            return "API Connected"
        return "API Error"
    except:
        return "Cannot connect to API. Please start the API server with: python run_api.py"


def get_available_models():
    """Fetch available models from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/list", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return [m.split('/')[-1] for m in models]  # Return short names
        return []
    except:
        return []


def get_available_datasets():
    """Fetch available datasets from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/datasets", timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []


def load_results(model_name, dataset_name):
    """Load and display results"""
    if not model_name or not dataset_name:
        gr.Info("Please select both model and dataset")
        return "", None
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/results/summary/{model_name}/{dataset_name}",
            timeout=10
        )
        
        if response.status_code != 200:
            gr.Warning(f"Error: {response.json().get('detail', 'Unknown error')}")
            return "", None
        
        summary = response.json()
        
        # Create summary table
        metrics_data = []
        for metric_name, metric_stats in summary['metrics'].items():
            metrics_data.append({
                'Metric': metric_name,
                'Mean': f"{metric_stats['mean']:.4f}",
                'Std': f"{metric_stats['std']:.4f}",
                'Min': f"{metric_stats['min']:.4f}",
                'Max': f"{metric_stats['max']:.4f}",
                'Count': metric_stats['count']
            })
        
        df = pd.DataFrame(metrics_data)
        
        summary_text = f"**Model:** {model_name}\n**Dataset:** {dataset_name}\n**Total Samples:** {summary['total_samples']}"
        
        return summary_text, df
        
    except Exception as e:
        gr.Warning(f"Error loading results: {str(e)}")
        return "", None


def view_sample(model_name, dataset_name, sample_idx):
    """View specific sample details"""
    if not model_name or not dataset_name:
        gr.Info("Please select model and dataset")
        return None
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/results/sample/{model_name}/{dataset_name}/{sample_idx}",
            timeout=10
        )
        
        if response.status_code != 200:
            gr.Warning(f"Error: {response.json().get('detail', 'Unknown error')}")
            return None
        
        result = response.json()
        return result['result']
        
    except Exception as e:
        gr.Warning(f"Error: {str(e)}")
        return None


def compare_models(selected_models, dataset_name):
    """Compare multiple models"""
    if not selected_models or len(selected_models) < 2:
        gr.Info("Please select at least 2 models")
        return None, None
    
    if not dataset_name:
        gr.Info("Please select a dataset")
        return None, None
    
    try:
        # Build query parameters
        params = {"data_name": dataset_name}
        for model in selected_models:
            params.setdefault("model_names", []).append(model)
        
        response = requests.get(
            f"{API_BASE_URL}/results/compare",
            params={"model_names": selected_models, "data_name": dataset_name},
            timeout=15
        )
        
        if response.status_code != 200:
            gr.Warning(f"Error: {response.json().get('detail', 'Unknown error')}")
            return None, None
        
        comparison_data = response.json()
        
        # Create comparison table
        metrics = ['SoS', 'UII', 'UEI', 'RS', 'RN']
        table_data = []
        
        for model in selected_models:
            if 'error' in comparison_data.get(model, {}):
                continue
            
            row = {'Model': model}
            for metric in metrics:
                if metric in comparison_data[model]:
                    mean = comparison_data[model][metric]['mean']
                    std = comparison_data[model][metric]['std']
                    row[metric] = f"{mean:.4f} ± {std:.4f}"
                else:
                    row[metric] = "N/A"
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Create radar chart
        fig = create_radar_chart(comparison_data, selected_models, metrics)
        
        return df, fig
        
    except Exception as e:
        gr.Warning(f"Error: {str(e)}")
        return None, None


def create_radar_chart(comparison_data, models, metrics):
    """Create radar chart for model comparison"""
    fig = go.Figure()
    
    for model in models:
        if 'error' in comparison_data.get(model, {}):
            continue
        
        values = []
        for metric in metrics:
            if metric in comparison_data[model]:
                values.append(comparison_data[model][metric]['mean'])
            else:
                values.append(0)
        
        # Close the radar chart
        values.append(values[0])
        metrics_closed = metrics + [metrics[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_closed,
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="ArC Metrics Comparison"
    )
    
    return fig


# Create Gradio interface
with gr.Blocks(title="ArC Dashboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Argument-based Consistency (ArC) Dashboard")
    gr.Markdown("Evaluate LLM toxicity explanations using ArC metrics")
    
    # API Status
    with gr.Row():
        api_status = gr.Textbox(
            label="API Status",
            value=check_api_health(),
            interactive=False,
            scale=3
        )
        refresh_btn = gr.Button("Refresh", scale=1)
        refresh_btn.click(fn=check_api_health, outputs=api_status)
    
    # Main tabs
    with gr.Tabs():
        # Tab 1: View Results
        with gr.Tab("View Results"):
            gr.Markdown("## View ArC Results")
            gr.Markdown("Select a model and dataset to view computed ArC metrics")
            
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Select Model",
                    choices=get_available_models(),
                    interactive=True
                )
                dataset_dropdown = gr.Dropdown(
                    label="Select Dataset",
                    choices=get_available_datasets(),
                    interactive=True
                )
            
            load_btn = gr.Button("Load Results", variant="primary")
            
            summary_text = gr.Markdown()
            summary_table = gr.Dataframe(label="Metrics Summary")
            
            gr.Markdown("### Sample Details")
            with gr.Row():
                sample_idx_input = gr.Number(
                    label="Sample Index",
                    value=0,
                    precision=0,
                    minimum=0
                )
                view_sample_btn = gr.Button("View Sample")
            
            sample_output = gr.JSON(label="Sample Result")
            
            # Event handlers
            load_btn.click(
                fn=load_results,
                inputs=[model_dropdown, dataset_dropdown],
                outputs=[summary_text, summary_table]
            )
            
            view_sample_btn.click(
                fn=view_sample,
                inputs=[model_dropdown, dataset_dropdown, sample_idx_input],
                outputs=sample_output
            )
        
        # Tab 2: Compare Models
        with gr.Tab("Compare Models"):
            gr.Markdown("## Compare Models")
            gr.Markdown("Select multiple models to compare their ArC metrics on the same dataset")
            
            with gr.Row():
                models_checkboxgroup = gr.CheckboxGroup(
                    label="Select Models to Compare",
                    choices=get_available_models(),
                    interactive=True
                )
                dataset_dropdown_compare = gr.Dropdown(
                    label="Select Dataset",
                    choices=get_available_datasets(),
                    interactive=True
                )
            
            compare_btn = gr.Button("Compare Models", variant="primary")
            
            comparison_table = gr.Dataframe(label="Metric Comparison")
            radar_plot = gr.Plot(label="Radar Chart")
            
            compare_btn.click(
                fn=compare_models,
                inputs=[models_checkboxgroup, dataset_dropdown_compare],
                outputs=[comparison_table, radar_plot]
            )
        
        # Tab 3: Run Pipeline
        with gr.Tab("Run Pipeline"):
            gr.Markdown("## Execute Full Pipeline")
            gr.Markdown("Generate LLM outputs, parse them, and compute ArC metrics")
            
            with gr.Row():
                pipeline_model = gr.Dropdown(
                    label="Model",
                    choices=get_available_models(),
                    interactive=True
                )
                pipeline_dataset = gr.Dropdown(
                    label="Dataset",
                    choices=get_available_datasets(),
                    interactive=True
                )
            
            with gr.Row():
                pipeline_batch_size = gr.Number(
                    label="Batch Size",
                    value=16,
                    precision=0
                )
                pipeline_data_size = gr.Number(
                    label="Data Size (optional)",
                    value=None,
                    precision=0
                )
            
            pipeline_explicit = gr.Checkbox(
                label="Explicit Prompting",
                value=True
            )
            
            run_pipeline_btn = gr.Button("Run Full Pipeline", variant="primary")
            pipeline_status = gr.Textbox(label="Status", interactive=False)
            
            def run_full_pipeline(model, dataset, batch_size, data_size, explicit):
                if not model or not dataset:
                    gr.Info("Please select model and dataset")
                    return "Please select model and dataset"
                
                try:
                    payload = {
                        "data_name": dataset,
                        "model_name": model,
                        "explicit_prompting": explicit,
                        "batch_size": int(batch_size),
                        "stages": ["initial", "internal", "external", "individual"]
                    }
                    if data_size:
                        payload["data_size"] = int(data_size)
                    
                    response = requests.post(
                        f"{API_BASE_URL}/pipeline/full",
                        json=payload,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        return f"Pipeline started successfully! This will take a while. Check logs for progress."
                    else:
                        gr.Warning(f"Error: {response.json().get('detail', 'Unknown error')}")
                        return f"Error: {response.json().get('detail', 'Unknown error')}"
                except Exception as e:
                    gr.Warning(f"Error: {str(e)}")
                    return f"Error: {str(e)}"
            
            run_pipeline_btn.click(
                fn=run_full_pipeline,
                inputs=[pipeline_model, pipeline_dataset, pipeline_batch_size, pipeline_data_size, pipeline_explicit],
                outputs=pipeline_status
            )
        
        # Tab 4: Configuration
        with gr.Tab("Configuration"):
            gr.Markdown("## Manage Configuration")
            gr.Markdown("View and edit configuration files with full CRUD operations")
            
            with gr.Tabs():
                # Models Configuration
                with gr.Tab("Models"):
                    gr.Markdown("### Model Size Configuration")
                    
                    with gr.Row():
                        load_models_btn = gr.Button("Load All Models", variant="secondary")
                        refresh_models_btn = gr.Button("Refresh", variant="secondary")
                    
                    models_table = gr.Dataframe(
                        headers=["Model Name", "Size"],
                        label="Current Models",
                        interactive=False
                    )
                    
                    gr.Markdown("#### Add/Edit Model")
                    with gr.Row():
                        model_name_input = gr.Textbox(label="Model Name", placeholder="e.g., meta-llama/Llama-3.1-8B-Instruct")
                        model_size_input = gr.Textbox(label="Size", placeholder="e.g., 8B")
                    
                    with gr.Row():
                        add_model_btn = gr.Button("Add New", variant="primary")
                        update_model_btn = gr.Button("Update Existing", variant="secondary")
                        delete_model_btn = gr.Button("Delete", variant="stop")
                    
                    model_status = gr.Textbox(label="Status", interactive=False)
                    
                    def load_models_table():
                        try:
                            response = requests.get(f"{API_BASE_URL}/config/models", timeout=5)
                            if response.status_code == 200:
                                config = response.json()
                                data = [[k, v] for k, v in config.items()]
                                return pd.DataFrame(data, columns=["Model Name", "Size"])
                            return None
                        except Exception as e:
                            gr.Warning(f"Error: {str(e)}")
                            return None
                    
                    def add_model(name, size):
                        if not name or not size:
                            return "Please provide both model name and size", load_models_table()
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/config/models",
                                json={"key": name, "value": size},
                                timeout=5
                            )
                            if response.status_code == 200:
                                return f"Model '{name}' added successfully", load_models_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_models_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_models_table()
                    
                    def update_model(name, size):
                        if not name or not size:
                            return "Please provide both model name and size", load_models_table()
                        try:
                            response = requests.put(
                                f"{API_BASE_URL}/config/models/{name}",
                                json={"value": size},
                                timeout=5
                            )
                            if response.status_code == 200:
                                return f"Model '{name}' updated successfully", load_models_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_models_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_models_table()
                    
                    def delete_model(name):
                        if not name:
                            return "Please provide model name", load_models_table()
                        try:
                            response = requests.delete(f"{API_BASE_URL}/config/models/{name}", timeout=5)
                            if response.status_code == 200:
                                return f"Model '{name}' deleted successfully", load_models_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_models_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_models_table()
                    
                    load_models_btn.click(fn=load_models_table, outputs=models_table)
                    refresh_models_btn.click(fn=load_models_table, outputs=models_table)
                    add_model_btn.click(fn=add_model, inputs=[model_name_input, model_size_input], outputs=[model_status, models_table])
                    update_model_btn.click(fn=update_model, inputs=[model_name_input, model_size_input], outputs=[model_status, models_table])
                    delete_model_btn.click(fn=delete_model, inputs=model_name_input, outputs=[model_status, models_table])
                
                # Datasets Configuration
                with gr.Tab("Datasets"):
                    gr.Markdown("### Dataset Path Configuration")
                    
                    with gr.Row():
                        load_datasets_btn = gr.Button("Load All Datasets", variant="secondary")
                        refresh_datasets_btn = gr.Button("Refresh", variant="secondary")
                    
                    datasets_table = gr.Dataframe(
                        headers=["Dataset Name", "Path"],
                        label="Current Datasets",
                        interactive=False
                    )
                    
                    gr.Markdown("#### Add/Edit Dataset")
                    with gr.Row():
                        dataset_name_input = gr.Textbox(label="Dataset Name", placeholder="e.g., civil_comments")
                        dataset_path_input = gr.Textbox(label="Path", placeholder="e.g., data/civil_comments.csv")
                    
                    with gr.Row():
                        add_dataset_btn = gr.Button("Add New", variant="primary")
                        update_dataset_btn = gr.Button("Update Existing", variant="secondary")
                        delete_dataset_btn = gr.Button("Delete", variant="stop")
                    
                    dataset_status = gr.Textbox(label="Status", interactive=False)
                    
                    def load_datasets_table():
                        try:
                            response = requests.get(f"{API_BASE_URL}/config/datasets", timeout=5)
                            if response.status_code == 200:
                                config = response.json()
                                data = [[k, v] for k, v in config.items()]
                                return pd.DataFrame(data, columns=["Dataset Name", "Path"])
                            return None
                        except Exception as e:
                            gr.Warning(f"Error: {str(e)}")
                            return None
                    
                    def add_dataset(name, path):
                        if not name or not path:
                            return "Please provide both dataset name and path", load_datasets_table()
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/config/datasets",
                                json={"key": name, "value": path},
                                timeout=5
                            )
                            if response.status_code == 200:
                                return f"Dataset '{name}' added successfully", load_datasets_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_datasets_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_datasets_table()
                    
                    def update_dataset(name, path):
                        if not name or not path:
                            return "Please provide both dataset name and path", load_datasets_table()
                        try:
                            response = requests.put(
                                f"{API_BASE_URL}/config/datasets/{name}",
                                json={"value": path},
                                timeout=5
                            )
                            if response.status_code == 200:
                                return f"Dataset '{name}' updated successfully", load_datasets_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_datasets_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_datasets_table()
                    
                    def delete_dataset(name):
                        if not name:
                            return "Please provide dataset name", load_datasets_table()
                        try:
                            response = requests.delete(f"{API_BASE_URL}/config/datasets/{name}", timeout=5)
                            if response.status_code == 200:
                                return f"Dataset '{name}' deleted successfully", load_datasets_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_datasets_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_datasets_table()
                    
                    load_datasets_btn.click(fn=load_datasets_table, outputs=datasets_table)
                    refresh_datasets_btn.click(fn=load_datasets_table, outputs=datasets_table)
                    add_dataset_btn.click(fn=add_dataset, inputs=[dataset_name_input, dataset_path_input], outputs=[dataset_status, datasets_table])
                    update_dataset_btn.click(fn=update_dataset, inputs=[dataset_name_input, dataset_path_input], outputs=[dataset_status, datasets_table])
                    delete_dataset_btn.click(fn=delete_dataset, inputs=dataset_name_input, outputs=[dataset_status, datasets_table])
                
                # Hyperparameters Configuration
                with gr.Tab("Hyperparameters"):
                    gr.Markdown("### ArC Hyperparameters")
                    
                    with gr.Row():
                        load_hyperparams_btn = gr.Button("Load All Hyperparameters", variant="secondary")
                        refresh_hyperparams_btn = gr.Button("Refresh", variant="secondary")
                    
                    hyperparams_table = gr.Dataframe(
                        headers=["Parameter Name", "Value"],
                        label="Current Hyperparameters",
                        interactive=False
                    )
                    
                    gr.Markdown("#### Add/Edit Hyperparameter")
                    with gr.Row():
                        hyperparam_name_input = gr.Textbox(label="Parameter Name", placeholder="e.g., threshold")
                        hyperparam_value_input = gr.Number(label="Value", value=0.0)
                    
                    with gr.Row():
                        add_hyperparam_btn = gr.Button("Add New", variant="primary")
                        update_hyperparam_btn = gr.Button("Update Existing", variant="secondary")
                        delete_hyperparam_btn = gr.Button("Delete", variant="stop")
                    
                    hyperparam_status = gr.Textbox(label="Status", interactive=False)
                    
                    def load_hyperparams_table():
                        try:
                            response = requests.get(f"{API_BASE_URL}/config/hyperparams", timeout=5)
                            if response.status_code == 200:
                                config = response.json()
                                data = [[k, v] for k, v in config.items()]
                                return pd.DataFrame(data, columns=["Parameter Name", "Value"])
                            return None
                        except Exception as e:
                            gr.Warning(f"Error: {str(e)}")
                            return None
                    
                    def add_hyperparam(name, value):
                        if not name or value is None:
                            return "Please provide both parameter name and value", load_hyperparams_table()
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/config/hyperparams",
                                json={"key": name, "value": value},
                                timeout=5
                            )
                            if response.status_code == 200:
                                return f"Hyperparameter '{name}' added successfully", load_hyperparams_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_hyperparams_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_hyperparams_table()
                    
                    def update_hyperparam(name, value):
                        if not name or value is None:
                            return "Please provide both parameter name and value", load_hyperparams_table()
                        try:
                            response = requests.put(
                                f"{API_BASE_URL}/config/hyperparams/{name}",
                                json={"value": value},
                                timeout=5
                            )
                            if response.status_code == 200:
                                return f"Hyperparameter '{name}' updated successfully", load_hyperparams_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_hyperparams_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_hyperparams_table()
                    
                    def delete_hyperparam(name):
                        if not name:
                            return "Please provide parameter name", load_hyperparams_table()
                        try:
                            response = requests.delete(f"{API_BASE_URL}/config/hyperparams/{name}", timeout=5)
                            if response.status_code == 200:
                                return f"Hyperparameter '{name}' deleted successfully", load_hyperparams_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_hyperparams_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_hyperparams_table()
                    
                    load_hyperparams_btn.click(fn=load_hyperparams_table, outputs=hyperparams_table)
                    refresh_hyperparams_btn.click(fn=load_hyperparams_table, outputs=hyperparams_table)
                    add_hyperparam_btn.click(fn=add_hyperparam, inputs=[hyperparam_name_input, hyperparam_value_input], outputs=[hyperparam_status, hyperparams_table])
                    update_hyperparam_btn.click(fn=update_hyperparam, inputs=[hyperparam_name_input, hyperparam_value_input], outputs=[hyperparam_status, hyperparams_table])
                    delete_hyperparam_btn.click(fn=delete_hyperparam, inputs=hyperparam_name_input, outputs=[hyperparam_status, hyperparams_table])
                
                # Prompts Configuration
                with gr.Tab("Prompts"):
                    gr.Markdown("### Prompt Instructions Configuration")
                    
                    with gr.Row():
                        load_prompts_btn = gr.Button("Load All Prompts", variant="secondary")
                        refresh_prompts_btn = gr.Button("Refresh", variant="secondary")
                    
                    prompts_table = gr.Dataframe(
                        headers=["Prompt Key", "Instruction"],
                        label="Current Prompts",
                        interactive=False
                    )
                    
                    gr.Markdown("#### Add/Edit Prompt")
                    with gr.Row():
                        prompt_key_input = gr.Textbox(label="Prompt Key", placeholder="e.g., initial_explicit")
                    
                    prompt_value_input = gr.Textbox(
                        label="Instruction",
                        placeholder="Enter prompt instruction text...",
                        lines=5
                    )
                    
                    with gr.Row():
                        add_prompt_btn = gr.Button("Add New", variant="primary")
                        update_prompt_btn = gr.Button("Update Existing", variant="secondary")
                        delete_prompt_btn = gr.Button("Delete", variant="stop")
                    
                    prompt_status = gr.Textbox(label="Status", interactive=False)
                    
                    def load_prompts_table():
                        try:
                            response = requests.get(f"{API_BASE_URL}/config/prompts", timeout=5)
                            if response.status_code == 200:
                                config = response.json()
                                data = [[k, str(v)[:100] + "..." if len(str(v)) > 100 else str(v)] for k, v in config.items()]
                                return pd.DataFrame(data, columns=["Prompt Key", "Instruction"])
                            return None
                        except Exception as e:
                            gr.Warning(f"Error: {str(e)}")
                            return None
                    
                    def add_prompt(key, value):
                        if not key or not value:
                            return "Please provide both prompt key and instruction", load_prompts_table()
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/config/prompts",
                                json={"key": key, "value": value},
                                timeout=5
                            )
                            if response.status_code == 200:
                                return f"Prompt '{key}' added successfully", load_prompts_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_prompts_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_prompts_table()
                    
                    def update_prompt(key, value):
                        if not key or not value:
                            return "Please provide both prompt key and instruction", load_prompts_table()
                        try:
                            response = requests.put(
                                f"{API_BASE_URL}/config/prompts/{key}",
                                json={"value": value},
                                timeout=5
                            )
                            if response.status_code == 200:
                                return f"Prompt '{key}' updated successfully", load_prompts_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_prompts_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_prompts_table()
                    
                    def delete_prompt(key):
                        if not key:
                            return "Please provide prompt key", load_prompts_table()
                        try:
                            response = requests.delete(f"{API_BASE_URL}/config/prompts/{key}", timeout=5)
                            if response.status_code == 200:
                                return f"Prompt '{key}' deleted successfully", load_prompts_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_prompts_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_prompts_table()
                    
                    load_prompts_btn.click(fn=load_prompts_table, outputs=prompts_table)
                    refresh_prompts_btn.click(fn=load_prompts_table, outputs=prompts_table)
                    add_prompt_btn.click(fn=add_prompt, inputs=[prompt_key_input, prompt_value_input], outputs=[prompt_status, prompts_table])
                    update_prompt_btn.click(fn=update_prompt, inputs=[prompt_key_input, prompt_value_input], outputs=[prompt_status, prompts_table])
                    delete_prompt_btn.click(fn=delete_prompt, inputs=prompt_key_input, outputs=[prompt_status, prompts_table])
                
                # Path Prefixes Configuration
                with gr.Tab("Paths"):
                    gr.Markdown("### Data Path Prefixes Configuration")
                    
                    with gr.Row():
                        load_paths_btn = gr.Button("Load All Paths", variant="secondary")
                        refresh_paths_btn = gr.Button("Refresh", variant="secondary")
                    
                    paths_table = gr.Dataframe(
                        headers=["Path Key", "Prefix"],
                        label="Current Path Prefixes",
                        interactive=False
                    )
                    
                    gr.Markdown("#### Add/Edit Path Prefix")
                    with gr.Row():
                        path_key_input = gr.Textbox(label="Path Key", placeholder="e.g., LLM_GENERATED_DATA_PREFIX")
                        path_value_input = gr.Textbox(label="Prefix", placeholder="e.g., llm_generated_data")
                    
                    with gr.Row():
                        add_path_btn = gr.Button("Add New", variant="primary")
                        update_path_btn = gr.Button("Update Existing", variant="secondary")
                        delete_path_btn = gr.Button("Delete", variant="stop")
                    
                    path_status = gr.Textbox(label="Status", interactive=False)
                    
                    def load_paths_table():
                        try:
                            response = requests.get(f"{API_BASE_URL}/config/paths", timeout=5)
                            if response.status_code == 200:
                                config = response.json()
                                data = [[k, v] for k, v in config.items()]
                                return pd.DataFrame(data, columns=["Path Key", "Prefix"])
                            return None
                        except Exception as e:
                            gr.Warning(f"Error: {str(e)}")
                            return None
                    
                    def add_path(key, value):
                        if not key or not value:
                            return "Please provide both path key and prefix", load_paths_table()
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/config/paths",
                                json={"key": key, "value": value},
                                timeout=5
                            )
                            if response.status_code == 200:
                                return f"Path '{key}' added successfully", load_paths_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_paths_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_paths_table()
                    
                    def update_path(key, value):
                        if not key or not value:
                            return "Please provide both path key and prefix", load_paths_table()
                        try:
                            response = requests.put(
                                f"{API_BASE_URL}/config/paths/{key}",
                                json={"value": value},
                                timeout=5
                            )
                            if response.status_code == 200:
                                return f"Path '{key}' updated successfully", load_paths_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_paths_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_paths_table()
                    
                    def delete_path(key):
                        if not key:
                            return "Please provide path key", load_paths_table()
                        try:
                            response = requests.delete(f"{API_BASE_URL}/config/paths/{key}", timeout=5)
                            if response.status_code == 200:
                                return f"Path '{key}' deleted successfully", load_paths_table()
                            return f"Error: {response.json().get('detail', 'Unknown error')}", load_paths_table()
                        except Exception as e:
                            return f"Error: {str(e)}", load_paths_table()
                    
                    load_paths_btn.click(fn=load_paths_table, outputs=paths_table)
                    refresh_paths_btn.click(fn=load_paths_table, outputs=paths_table)
                    add_path_btn.click(fn=add_path, inputs=[path_key_input, path_value_input], outputs=[path_status, paths_table])
                    update_path_btn.click(fn=update_path, inputs=[path_key_input, path_value_input], outputs=[path_status, paths_table])
                    delete_path_btn.click(fn=delete_path, inputs=path_key_input, outputs=[path_status, paths_table])


if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )

Argument-based Consistency in Toxicity Explanations of LLMs
===========================================================

.. image:: https://github.com/uofthcdslab/ArC/blob/main/utils/ArC_intro.png
  :align: center
  :width: 400px

The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' *reasoning* about toxicity---from their explanations that justify a stance---to enhance their trustworthiness in downstream tasks. In our recent `paper <https://arxiv.org/pdf/2506.19113>`_, we propose a novel, theoretically-grounded multi-dimensional criterion, **Argument-based Consistency (ArC)**, that measures how LLMs' free-form toxicity explanations reflect those of a rational human under ideal conditions.
We develop six metrics, based on uncertainty quantification, to comprehensively evaluate ArC of LLMs' toxicity explanations with no human involvement, and highlight how "non-ideal" the explanations are. This repository contains the code and sample data to reproduce our results. 

The complete LLM-generated toxicity explanations and our ArC scores are available on `Hugging Face <https://huggingface.co/collections/uofthcdslab/ArC-6857895ac09959da821bd015>`_. The complete LLM output tokens and entropy scores are available upon request.


Requirements:
=============

``pip install -r requirements.txt``


Quick Start:
============

**1. Compute ArC Metrics:**

``python arc.py``

This computes ArC metrics for all models and datasets using default parameters (explicit prompting enabled, logits-based entropy), storing results in ``arc_results/``

**2. View Results (Command Line):**

``python view_results.py --model_name Llama-3.1-8B-Instruct --data_name civil_comments``

View summary statistics for a specific model/dataset combination.

**3. Explore Results (Jupyter Notebook):**

See ``demo_arc_metrics.ipynb`` for an interactive demonstration of how to use the ArC classes and visualize metric outputs.


Pipeline:
=========

Quick Demo (with sample data):
------------------------------

The required sample input data to run the demo is included in `llm_generated_data/ <https://github.com/uofthcdslab/ArC/tree/main/llm_generated_data>`_ and `parsed_data/ <https://github.com/uofthcdslab/ArC/tree/main/parsed_data>`_ directories. To compute ArC metrics on this sample data, run:

``python arc.py``

This will compute the ArC metrics for the sample data and store the results in `arc_results/ <https://github.com/uofthcdslab/ArC/tree/main/arc_results>`_ directory. The results include ArC scores for different models and datasets.

**Code Architecture:**

The codebase uses a clean architecture that separates concerns:

- ``core/metrics/``: Individual metric implementations (SoS, DiS, UII, UEI, RS, RN)
- ``core/processors/``: Confidence and similarity computation processors
- ``core/models/``: Configuration and data models
- ``services/``: High-level ArC computation service


Reproducing Full Pipeline:
--------------------------

**Using an existing or a new dataset:**

1. Add the dataset name and path in `utils/data_path_map.json <https://github.com/uofthcdslab/ArC/blob/main/utils/data_path_map.json>`_.
2. Include the main processing function for the dataset in `utils/data_processor.py <https://github.com/uofthcdslab/ArC/blob/main/utils/data_processor.py>`_ and give it the exact same name as the dataset.
3. Access shared parameters and methods defined in the `DataLoader <https://github.com/uofthcdslab/ArC/blob/main/data_loader.py#L8>`_ class in `data_loader.py <https://github.com/uofthcdslab/ArC/blob/main/data_loader>`_ through instance references.


**LLM explanation generation and parsing:**

In the paper, we describe a three-stage pipeline to compute **ArC** metrics. The pipeline consists of:

1. Stage **JUSTIFY** where LLMs generate explanations for their toxicity decisions (denoted by ``stage="initial"``).
2. Stage **UPHOLD-REASON** where LLMs generate post-hoc explanations to assess the sufficiency of reasons provided in the **JUSTIFY** stage (denoted by ``stage="internal"`` or ``stage="external"``).
3. Stage **UPHOLD-STACE** where LLMs generate post-hoc explanations to assess the sufficiency and necessity of individual reasons of **JUSTIFY** stage (denoted by ``stage="individual"``).

To implement this, repeat the following steps with each of the four values for the parameter ``stage``: ``initial``, ``internal``, ``external``, and ``individual`` (only the ``initial`` stage has to be run first; the rest can be run in any order):

1. Run `generate.py <https://github.com/uofthcdslab/ArC/blob/main/generate.py>`_ with ``--generation_stage=initial/internal/external/individual`` and other optional changes to the generation hyperparameters. 
2. LLM outputs (tokens, token entropies, and texts) will be generated and stored in ``llm_generated_data/<model_name>/<data_name>/<stage>``. 
3. Run `parse.py <https://github.com/uofthcdslab/ArC/blob/main/parse.py>`_ with ``stage=initial/internal/external/individual`` and other optional parameters to extract LLM decisions, reasons, and other relevant information for computing ArC.
4. The parsed outputs will be stored in ``parsed_data/<model_name>/<data_name>/<stage>``.


**Computing ArC metrics:**

1. Run `arc.py <https://github.com/uofthcdslab/ArC/blob/main/arc.py>`_ with optional parameters to compute ArC metrics for all combinations of models and datasets.

Supported parameters:
- ``--explicit_prompting``: Use explicit prompting (True/False, default: True)
- ``--use_scores``: Use entropy of scores instead of logits (True/False, default: False)
- ``--similarity_model``: Semantic similarity model name (default: cross-encoder/stsb-distilroberta-base)

2. The outputs will be computed for each sample instance and stored in ``arc_results/<model_name>/<data_name>/<sample_index>.pkl``.


Viewing Results:
----------------

After computing ArC metrics, use `view_results.py <https://github.com/uofthcdslab/ArC/blob/main/view_results.py>`_ to analyze and summarize the results:

**View summary statistics for all samples:**

``python view_results.py --model_name Llama-3.1-8B-Instruct --data_name civil_comments``

This displays aggregated statistics (mean, std, min, max) for all ArC metrics:

- **Relevance Dimension:** SoS (Sufficiency of Stance), DiS-DPP, DiS-Avg (Diversity of Stance)
- **Internal Reliance Dimension:** UII (Uncertainty in Internal Informativeness), Internal Δ-PE
- **External Reliance Dimension:** UEI (Uncertainty in External Informativeness), External Δ-PE
- **Individual Reliance Dimension:** RS (Reason Sufficiency), RN (Reason Necessity)
- **Decision Confidence:** Initial, Internal, and External decision confidences

**View detailed results for a specific sample:**

``python view_results.py --model_name Llama-3.1-8B-Instruct --data_name civil_comments --sample_idx 0``

**List all available results:**

``python view_results.py``

This will show all computed models and datasets with their sample counts.

**Available options:**

- ``--model_name``: Model name (e.g., Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, Llama-3.3-70B-Instruct, Ministral-8B-Instruct-2410)
- ``--data_name``: Dataset name (e.g., civil_comments, hate_explain, implicit_toxicity, real_toxicity_prompts, toxigen)
- ``--sample_idx``: Specific sample index to view detailed results (optional)
- ``--results_path``: Path to results directory (default: arc_results)


Roadmap:
========
1. We are working on updating the parser files to support more datasets and models. We will soon integrate the results of Microsoft Phi-4 reasoning model.
2. We will include the results of naive prompting without explicit reasoning instructions.


Citing:
=======
Bibtex::

	@article{mothilal2025ArC,
	  title={Argument-based Consistency in Toxicity Explanations of LLMs},
	  author={K Mothilal, Ramaravind and Roy, Joanna and Ahmed, Syed Ishtiaque and Guha, Shion},
	  journal={arXiv preprint arXiv:2506.19113},
	  year={2025}
	}

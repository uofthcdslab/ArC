
Argument-Based Consistency in Toxicity Explanations of LLMs
===========================================================

.. image:: https://github.com/uofthcdslab/ArC/blob/main/utils/arc_overall_flow.png
  :align: center
  :width: 400px

The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' *reasoning* about toxicity---from their explanations that justify a stance---to enhance their trustworthiness in downstream tasks. Despite extensive research on explainability, it is not straightforward to adopt existing methods to evaluate free-form toxicity explanation due to their over-reliance on input text perturbations, among other challenges. To account for these, in our recent `paper <https://arxiv.org/pdf/2506.19113>`_, we propose a novel, theoretically-grounded multi-dimensional criterion, **Argument-based Consistency** (ArC), that measures the extent to which LLMs' free-form toxicity explanations reflect an ideal and logical argumentation process. We develop six metrics, based on uncertainty quantification, to comprehensively evaluate ArC of LLMs' toxicity explanations with no human involvement, and highlight how “non-ideal” the explanations are. Our results show that while LLMs generate plausible explanations to simple prompts, their reasoning about toxicity breaks down when prompted about the nuanced relations between the complete set of reasons, the individual reasons, and their toxicity stances, resulting in inconsistent and irrelevant responses. This repository contains the code and sample data to reproduce our results. 

The complete LLM-generated toxicity explanations and our ArC scores are available on `Hugging Face <https://huggingface.co/collections/uofthcdslab/arc>`_. The complete LLM output tokens and entropy scores are available upon request.


Requirements:
=============

``pip install -r requirements.txt``


Pipeline:
=========

Quick Demo (with sample data):
------------------------------

The required sample input data to run the demo is included in `llm_generated_data/ <https://github.com/uofthcdslab/ArC/tree/main/llm_generated_data>`_ and `parsed_data/ <https://github.com/uofthcdslab/ArC/tree/main/parsed_data>`_ directories. To compute ArC metrics on this sample data, run the following command:

``python haf.py``

This will compute the ArC metrics for the sample data and store the results in `haf_results/ <https://github.com/uofthcdslab/ArC/tree/main/haf_results>`_ directory. The results include ArC scores for different models and datasets.


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

1. Run `haf.py <https://github.com/uofthcdslab/ArC/blob/main/haf.py>`_ with optional parameters to compute ArC metrics for all combinations of models and datasets.
2. The outputs will be computed for each sample instance and stored in ``haf_results/<model_name>/<data_name>/<sample_index>.pkl``.


Roadmap:
========
1. We are working on updating the parser files to support more datasets and models. We will soon integrate the results of Microsoft Phi-4 reasoning model.
2. We will include the results of naive prompting without explicit reasoning instructions.


Citing:
=======
Bibtex::

	@article{mothilal2025haf,
	  title={Human-Aligned Faithfulness in Toxicity Explanations of LLMs},
	  author={K Mothilal, Ramaravind and Roy, Joanna and Ahmed, Syed Ishtiaque and Guha, Shion},
	  journal={arXiv preprint arXiv:2506.19113},
	  year={2025}
	}

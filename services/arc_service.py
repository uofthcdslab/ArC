"""Main ArC (Argument-based Consistency) computation service"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer

from core.models.arc_config import ArCConfig
from core.metrics import SoSMetric, DiSMetric, UIIMetric, UEIMetric, RSMetric, RNMetric
from core.processors import ConfidenceProcessor, SimilarityProcessor
from utils import helpers as hp
from utils.logger_setup import setup_logger


TARGET_SENTS = {
    'YES': ['yes additional reasons are required', 'there are additional reasons', 'provided reasons are insufficient'],
    'NO': ['no additional reasons are required', 'additional reasons are not required', 'there are no additional reasons', 'provided reasons are sufficient'],
}


class ArCService:
    """Main service for ArC computation"""
    
    def __init__(self, config: ArCConfig):
        """
        Initialize ArC service
        
        Args:
            config: ArC configuration
        """
        self.config = config
        self.logger = setup_logger("arc_service", "ERROR", "arc_service_logs")
        
        # Load model and data details
        with open("utils/model_size_map.json", "r") as file:
            model_size = json.load(file)
        with open("utils/data_path_map.json", "r") as file:
            data_path = json.load(file)
        
        self.data_names = list(data_path.keys())
        self.model_names = list(model_size.keys())
        self.tokenizers_dict = {}
        
        # Initialize processors
        self.confidence_processor = ConfidenceProcessor(self.logger)
        self.similarity_processor = SimilarityProcessor(config.similarity_model, self.logger)
        
        # Decision importance maps for individual metrics
        self.individual_decision_imp = {
            'RS': {'NO': 1.0, 'MAYBE': 0.5, 'YES': 0.1, 'NO OR UNCLEAR DECISION': 0.1},
            'RN': {'YES': 1.0, 'MAYBE': 0.5, 'NO': 0.1, 'NO OR UNCLEAR DECISION': 0.1}
        }
        
        # Initialize metrics
        self.sos_metric = SoSMetric(self.logger)
        self.dis_metric = DiSMetric(self.logger)
        self.uii_metric = UIIMetric(self.similarity_processor, self.logger)
        self.uei_metric = UEIMetric(self.similarity_processor, self.logger)
        self.rs_metric = RSMetric(self.similarity_processor, self.individual_decision_imp['RS'], self.logger)
        self.rn_metric = RNMetric(self.similarity_processor, self.individual_decision_imp['RN'], self.logger)
    
    def get_tokenizer(self, model_name: str):
        """Get or load tokenizer for model"""
        if model_name not in self.tokenizers_dict:
            self.tokenizers_dict[model_name] = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizers_dict[model_name]
    
    def get_new_decision(self, decision_sent: str) -> str:
        """
        Determine decision type from decision sentence
        
        Args:
            decision_sent: Decision sentence text
            
        Returns:
            Decision type: 'YES', 'NO', or 'NO OR UNCLEAR DECISION'
        """
        prob_yes = max([float(self.similarity_processor.sims_hp.predict([(decision_sent, TARGET_SENTS['YES'][i])])[0]) 
                       for i in range(len(TARGET_SENTS['YES']))])
        prob_no = max([float(self.similarity_processor.sims_hp.predict([(decision_sent, TARGET_SENTS['NO'][i])])[0]) 
                      for i in range(len(TARGET_SENTS['NO']))])
        
        if prob_yes < 0.15 and prob_no < 0.15:
            return 'NO OR UNCLEAR DECISION'
        else:
            if prob_yes >= prob_no:
                return 'YES'
            else:
                return 'NO'
    
    def compute_sample(self, sample_ix: int, model_name: str, data_name: str,
                      output_tokens_dict: Dict, parsed_output_dict: Dict) -> Dict:
        """
        Compute all ArC metrics for a single sample
        
        Args:
            sample_ix: Sample index
            model_name: Model name
            data_name: Dataset name
            output_tokens_dict: Dictionary of output tokens
            parsed_output_dict: Dictionary of parsed outputs
            
        Returns:
            Dictionary containing all computed metrics
        """
        tokenizer = self.get_tokenizer(model_name)
        result = {}
        
        # ===== INITIAL STAGE: Relevance Dimension =====
        result.update(self._compute_initial_stage(
            sample_ix, tokenizer, output_tokens_dict, parsed_output_dict,
            model_name, data_name
        ))
        
        if len(result.get('initial_reasons', [])) == 0:
            self.logger.warning(f"No reasons found for sample {sample_ix}")
            return result
        
        # ===== RELIANCE STAGES: Internal and External =====
        for reliance_type, metric in [('internal', self.uii_metric), ('external', self.uei_metric)]:
            reliance_result = self._compute_reliance_stage(
                sample_ix, tokenizer, reliance_type, metric,
                output_tokens_dict, parsed_output_dict, result,
                model_name, data_name
            )
            result.update(reliance_result)
        
        # ===== INDIVIDUAL STAGE =====
        if self.config.explicit_prompting != '':
            individual_result = self._compute_individual_stage(
                sample_ix, tokenizer, output_tokens_dict, parsed_output_dict, result,
                model_name, data_name
            )
            result.update(individual_result)
        
        return result
    
    def _compute_initial_stage(self, sample_ix: int, tokenizer, 
                               output_tokens_dict: Dict, parsed_output_dict: Dict,
                               model_name: str = None, data_name: str = None) -> Dict:
        """Compute initial stage metrics (SoS, DiS)"""
        result = {}
        
        # Decision sentence confidence
        decision_sent = parsed_output_dict['initial']['decision_sentences'][sample_ix]
        decision_sent_tokens = tokenizer(decision_sent, add_special_tokens=False)['input_ids']
        start_ix, end_ix = self.confidence_processor.get_indices(
            torch.tensor(decision_sent_tokens),
            output_tokens_dict['initial'][sample_ix]
        )
        out_tokens = output_tokens_dict['initial'][sample_ix][start_ix:end_ix].tolist()
        confidence, _ = self.confidence_processor.compute_confidence(
            start_ix, out_tokens, decision_sent_tokens,
            parsed_output_dict['initial']['entropies_' + self.config.entropy_mode][sample_ix],
            parsed_output_dict['initial']['decision_relevances'][sample_ix]
        )
        result['initial_decision_confidence'] = confidence
        
        # Extract initial reasons
        initial_reasons = parsed_output_dict['initial']['reasons'][sample_ix]
        result['initial_reasons'] = initial_reasons
        result['initial_token_mismatch'] = []
        result['initial_reasons_confidences'] = []
        
        reasons_tokens = tokenizer(initial_reasons, add_special_tokens=False)['input_ids']
        initial_reasons_sims_input = parsed_output_dict['initial']['sims_input'][sample_ix]
        initial_reasons_sims_reasons = parsed_output_dict['initial']['sims_reasons'][sample_ix]
        
        # Compute confidence for each reason
        for reason_ix in range(len(initial_reasons)):
            start_ix, end_ix = parsed_output_dict['initial']['reasons_indices'][sample_ix][reason_ix]
            out_tokens = output_tokens_dict['initial'][sample_ix][start_ix:end_ix].tolist()
            confidence, encoding_issue = self.confidence_processor.compute_confidence(
                start_ix, out_tokens, reasons_tokens[reason_ix],
                parsed_output_dict['initial']['entropies_' + self.config.entropy_mode][sample_ix],
                parsed_output_dict['initial']['reasons_relevances'][sample_ix][reason_ix]
            )
            result['initial_reasons_confidences'].append(confidence)
            if encoding_issue:
                self.logger.warning(f"Encoding issue: {model_name}, {data_name}, initial, sample {sample_ix}, reason {reason_ix}")
                result['initial_token_mismatch'].append(reason_ix)
        
        # Compute SoS
        sos_data = {
            'initial_reasons_confidences': result['initial_reasons_confidences'],
            'initial_reasons_sims_input': initial_reasons_sims_input
        }
        result['SoS'] = self.sos_metric.compute(sos_data)
        
        # Compute DiS
        dis_data = {
            'initial_reasons': initial_reasons,
            'initial_reasons_confidences': result['initial_reasons_confidences'],
            'initial_reasons_sims_reasons': initial_reasons_sims_reasons
        }
        dis_result = self.dis_metric.compute(dis_data)
        result.update(dis_result)
        
        return result
    
    def _compute_reliance_stage(self, sample_ix: int, tokenizer, reliance_type: str,
                               metric, output_tokens_dict: Dict, 
                               parsed_output_dict: Dict, initial_result: Dict,
                               model_name: str = None, data_name: str = None) -> Dict:
        """Compute reliance stage metrics (UII or UEI)"""
        result = {}
        
        # Decision sentence confidence
        decision_sent = parsed_output_dict[reliance_type]['decision_sentences'][sample_ix]
        decision_sent_tokens = tokenizer(decision_sent, add_special_tokens=False)['input_ids']
        start_ix, end_ix = self.confidence_processor.get_indices(
            torch.tensor(decision_sent_tokens),
            output_tokens_dict[reliance_type][sample_ix]
        )
        out_tokens = output_tokens_dict[reliance_type][sample_ix][start_ix:end_ix].tolist()
        confidence, _ = self.confidence_processor.compute_confidence(
            start_ix, out_tokens, decision_sent_tokens,
            parsed_output_dict[reliance_type]['entropies_' + self.config.entropy_mode][sample_ix],
            parsed_output_dict[reliance_type]['decision_relevances'][sample_ix]
        )
        result[f'{reliance_type}_decision_confidence'] = confidence
        
        # Extract reliance reasons
        reliance_reasons = parsed_output_dict[reliance_type]['reasons'][sample_ix]
        
        if len(reliance_reasons) == 0:
            self.logger.warning(f"No reasons found for {reliance_type} stage, sample {sample_ix}")
            return result
        
        result[f'{reliance_type}_token_mismatch'] = []
        result[f'{reliance_type}_reasons_confidences'] = []
        reasons_tokens = tokenizer(reliance_reasons, add_special_tokens=False)['input_ids']
        
        # Compute confidence for each reason
        for reason_ix in range(len(reliance_reasons)):
            start_ix, end_ix = parsed_output_dict[reliance_type]['reasons_indices'][sample_ix][reason_ix]
            out_tokens = output_tokens_dict[reliance_type][sample_ix][start_ix:end_ix].tolist()
            confidence, encoding_issue = self.confidence_processor.compute_confidence(
                start_ix, out_tokens, reasons_tokens[reason_ix],
                parsed_output_dict[reliance_type]['entropies_' + self.config.entropy_mode][sample_ix],
                parsed_output_dict[reliance_type]['reasons_relevances'][sample_ix][reason_ix]
            )
            result[f'{reliance_type}_reasons_confidences'].append(confidence)
            if encoding_issue:
                self.logger.warning(f"Encoding issue: {model_name}, {data_name}, {reliance_type}, sample {sample_ix}, reason {reason_ix}")
                result[f'{reliance_type}_token_mismatch'].append(reason_ix)
        
        # Compute metric (UII or UEI)
        metric_data = {
            f'{reliance_type}_reasons': reliance_reasons,
            f'{reliance_type}_reasons_confidences': result[f'{reliance_type}_reasons_confidences'],
            'initial_reasons': initial_result['initial_reasons'],
            'initial_reasons_confidences': initial_result['initial_reasons_confidences']
        }
        metric_result = metric.compute(metric_data)
        result[metric.get_name()] = metric_result
        
        return result
    
    def _compute_individual_stage(self, sample_ix: int, tokenizer,
                                 output_tokens_dict: Dict, parsed_output_dict: Dict,
                                 initial_result: Dict,
                                 model_name: str = None, data_name: str = None) -> Dict:
        """Compute individual stage metrics (RS or RN)"""
        result = {}
        
        # Check if individual data exists
        if sample_ix >= len(output_tokens_dict['individual']):
            self.logger.warning(f"No individual data for sample {sample_ix}")
            return result
        
        if len(output_tokens_dict['individual'][sample_ix]) == 0:
            self.logger.warning(f"Empty individual data for sample {sample_ix}")
            return result
        
        if parsed_output_dict['initial']['decisions'][sample_ix] == 'NO OR UNCLEAR DECISION':
            self.logger.warning(f"No clear decision for sample {sample_ix}")
            return result
        
        # Determine metric type based on initial decision
        if parsed_output_dict['initial']['decisions'][sample_ix] == 'non-toxic':
            metric = self.rn_metric
        else:
            metric = self.rs_metric
        
        result['individual_token_mismatch'] = {}
        result['individual_reasons_confidences'] = {}
        result['individual_decision_confidence'] = {}
        reliance_reasons = parsed_output_dict['individual']['reasons'][sample_ix]
        new_decisions = []
        
        # Process each subsample
        for subsample_ix in range(len(output_tokens_dict['individual'][sample_ix])):
            # Get new decision
            decision_sent = parsed_output_dict['individual']['decision_sentences'][sample_ix][subsample_ix]
            new_decision = self.get_new_decision(decision_sent)
            new_decisions.append(new_decision)
            
            # Decision confidence
            decision_sent_tokens = tokenizer(decision_sent, add_special_tokens=False)['input_ids']
            start_ix, end_ix = self.confidence_processor.get_indices(
                torch.tensor(decision_sent_tokens),
                output_tokens_dict['individual'][sample_ix][subsample_ix]
            )
            out_tokens = output_tokens_dict['individual'][sample_ix][subsample_ix][start_ix:end_ix].tolist()
            confidence, _ = self.confidence_processor.compute_confidence(
                start_ix, out_tokens, decision_sent_tokens,
                parsed_output_dict['individual']['entropies_' + self.config.entropy_mode][sample_ix][subsample_ix],
                parsed_output_dict['individual']['decision_relevances'][sample_ix][subsample_ix]
            )
            result['individual_decision_confidence'][subsample_ix] = confidence
            
            # Reason confidences
            if len(reliance_reasons[subsample_ix]) == 0:
                result['individual_reasons_confidences'][subsample_ix] = []
                result['individual_token_mismatch'][subsample_ix] = []
                continue
            
            result['individual_token_mismatch'][subsample_ix] = []
            result['individual_reasons_confidences'][subsample_ix] = []
            reasons_tokens = tokenizer(reliance_reasons[subsample_ix], add_special_tokens=False)['input_ids']
            
            for reason_ix in range(len(reliance_reasons[subsample_ix])):
                start_ix, end_ix = parsed_output_dict['individual']['reasons_indices'][sample_ix][subsample_ix][reason_ix]
                out_tokens = output_tokens_dict['individual'][sample_ix][subsample_ix][start_ix:end_ix].tolist()
                confidence, encoding_issue = self.confidence_processor.compute_confidence(
                    start_ix, out_tokens, reasons_tokens[reason_ix],
                    parsed_output_dict['individual']['entropies_' + self.config.entropy_mode][sample_ix][subsample_ix],
                    parsed_output_dict['individual']['reasons_relevances'][sample_ix][subsample_ix][reason_ix]
                )
                result['individual_reasons_confidences'][subsample_ix].append(confidence)
                if encoding_issue:
                    self.logger.warning(f"Encoding issue: {model_name}, {data_name}, individual, sample {sample_ix}, subsample {subsample_ix}, reason {reason_ix}")
                    result['individual_token_mismatch'][subsample_ix].append(reason_ix)
        
        # Compute RS or RN metric
        metric_data = {
            'individual_reasons': reliance_reasons,
            'individual_reasons_confidences': result['individual_reasons_confidences'],
            'individual_decision_confidence': result['individual_decision_confidence'],
            'new_decisions': new_decisions,
            'initial_reasons': initial_result['initial_reasons'],
            'initial_reasons_confidences': initial_result['initial_reasons_confidences']
        }
        result[metric.get_name()] = metric.compute(metric_data)
        
        return result
    
    # ===== UNUSED METHODS (Kept for reference, commented out) =====
    # These methods are from the original code but are not used in the computation flow
    
    # def find_individual_decision(self, model_name, output_tokens):
    #     """
    #     Find individual decisions - the below adjustment is required only for a few samples 
    #     where conflicting decisions are generated - for others this is skewing the interpretation
    #     """
    #     import re
    #     
    #     decision = []  # "NO OR UNCLEAR DECISION"
    #     decision_sentence = []  # "NO OR UNCLEAR DECISION"
    #     decision_indices = []  # (0,0)
    #     decision_tokens = []  # []
    #     decision_relevances = []  # []
    #     
    #     match_words = ['may', 'might', 'could', 'but', 'however', 'though', 'although']
    #     tokenizer = self.get_tokenizer(model_name)
    #     
    #     for batch_ix in range(len(output_tokens)):  # for each batch of a sample
    #         output_text = tokenizer.decode(output_tokens[batch_ix])
    #         sentences = re.split(r'(?<=[.!?])\s+|\n+', output_text.strip()) or [""]
    #         decision_found = False
    #         
    #         for sent in sentences[0:2]:
    #             prob_yes = max([float(self.similarity_processor.sims_hp.predict([(sent, TARGET_SENTS['YES'][i])])[0]) 
    #                            for i in range(len(TARGET_SENTS['YES']))])
    #             prob_no = max([float(self.similarity_processor.sims_hp.predict([(sent, TARGET_SENTS['NO'][i])])[0]) 
    #                           for i in range(len(TARGET_SENTS['NO']))])
    #             
    #             if prob_yes < 0.15 and prob_no < 0.15:
    #                 continue  # check the next sentence
    #             
    #             decision_found = True
    #             decision_sentence.append(sent)  # if at least one prob is > 0.33, then it has alternative decision
    #             if re.search(r"(" + "|".join(match_words) + ")", sent, re.IGNORECASE):
    #                 decision.append('MAYBE')
    #             elif prob_yes >= prob_no:
    #                 decision.append('YES')
    #             else:
    #                 decision.append('NO')
    #             break
    #         
    #         if not decision_found:
    #             decision.append('NO OR UNCLEAR DECISION')
    #             decision_sentence.append('NO OR UNCLEAR DECISION')
    #             decision_tokens.append([])
    #             decision_indices.append((0, 0))
    #             decision_relevances.append([])
    #             continue
    #         
    #         decision_sent_tokens = tokenizer(decision_sentence[batch_ix], add_special_tokens=False)['input_ids']
    #         decision_tokens.append(decision_sent_tokens)
    #         start_ix, end_ix = self.confidence_processor.get_indices(torch.tensor(decision_sent_tokens), output_tokens[batch_ix])
    #         decision_indices.append((start_ix, end_ix))
    #         rels = self.get_relevance_scores_for_sentence(model_name, torch.tensor(decision_sent_tokens), decision_sentence[batch_ix])
    #         decision_relevances.append(rels)
    #     
    #     return decision, decision_sentence, decision_tokens, decision_indices, decision_relevances
    
    # def get_relevance_scores_for_sentence(self, model_name, sentence_tokens, sentence_target_str):
    #     """Get relevance scores for a sentence by masking each token"""
    #     tokenizer = self.get_tokenizer(model_name)
    #     sentence_tokens_masked = [sentence_tokens[torch.arange(len(sentence_tokens)) != i] for i in range(len(sentence_tokens))]
    #     sentence_str_masked = tokenizer.batch_decode(sentence_tokens_masked)
    #     sentence_pairs = [(sentence_target_str, sentence_m) for sentence_m in sentence_str_masked]
    #     scores = self.similarity_processor.sims_hp.predict(sentence_pairs)
    #     return [float(1 - s) for s in scores]
    
    # def store_individual_decisions_info(self, sample_ix, model_name, data_name, ind_decision, 
    #                                    ind_decision_sent, ind_decision_tokens, ind_decision_indices, 
    #                                    ind_decision_relevances):
    #     """Store individual decision information to file"""
    #     from utils.data_path_prefixes import ARC_RESULTS_PATH
    #     import pickle
    #     
    #     directory_path = Path(ARC_RESULTS_PATH + "/" + model_name.split('/')[1] + '/' + data_name + '/' + 'individual_decisions/')
    #     directory_path.mkdir(parents=True, exist_ok=True)
    #     file_path = directory_path / (str(sample_ix) + '.pkl')
    #     self.logger.info(f"💾 Saving results to {file_path}")
    #     results = {
    #         'ind_decision': ind_decision,
    #         'ind_decision_sent': ind_decision_sent,
    #         'ind_decision_tokens': ind_decision_tokens,
    #         'ind_decision_indices': ind_decision_indices,
    #         'ind_decision_relevances': ind_decision_relevances
    #     }
    #     with file_path.open("wb") as f:
    #         pickle.dump(results, f)

import types
from utils import data_processor
import json
import pickle
from pathlib import Path
from utils.data_path_prefixes import PARSE_OUTPUT_PATH

class DataLoader:
    """A class to load input data based on user input."""

    def __init__(self, data_name,
                 total_samples=10000,
                 random_state=17,
                 batch_size = 16,
                 save_processed_sampled_data=True,
                 select_new_dataset_samples=False,
                 toxicity_type='toxic'):
        """Init method

        :data_name: Name as stored in utils/data_path_map.
        :total_samples: Total samples to be loaded. Defaults to 10,000.
        :random_state: Random state for sampling. Defaults to 17.
        :save_processed_sampled_data: Save processed input data for feeding it to LMs. Defaults to False.
        :toxicity_type: Type of samples to load - 'toxic', 'nontoxic', or 'both'. Defaults to 'toxic'.
        """
        self.data_name = data_name
        self.total_samples = min(total_samples, 10000)
        self.random_state = random_state
        self.batch_size = batch_size
        self.save_processed_sampled_data = save_processed_sampled_data
        self.select_new_dataset_samples = select_new_dataset_samples
        self.toxicity_type = toxicity_type

        # Handle nontoxic data loading
        base_data_name = data_name.replace('_nontoxic', '')
        func = getattr(data_processor, base_data_name)
        self.data_processing_func = types.MethodType(func, self)

    def load_for_initial_generation(self, **kwargs):
        """Load data based on data_name."""

        # Check if loading nontoxic samples from pre-downloaded CSV
        if self.toxicity_type == 'nontoxic' or '_nontoxic' in self.data_name:
            base_name = self.data_name.replace('_nontoxic', '')
            nontoxic_path = Path(PARSE_OUTPUT_PATH).parent / 'processed_sampled_input_data' / f'nontoxic_{base_name}.csv'

            if nontoxic_path.exists():
                import pandas as pd
                data = pd.read_csv(nontoxic_path)
            else:
                raise FileNotFoundError(f"Non-toxic samples not found at {nontoxic_path}. Run download_nontoxic_samples.py first.")
        else:
            with open("utils/data_path_map.json", "r") as file:
                data_path_map = json.load(file)

            base_name = self.data_name.replace('_nontoxic', '')
            data_path = data_path_map[base_name]
            data = self.data_processing_func(data_path, **kwargs)

        # add batch numbers - current method is just based on length
        # TODO: do this effectively with DP or clustering
        data['text_len'] = data['input_text'].apply(len)
        data = data.sort_values('text_len')
        # num_batches = int(np.ceil(len(data) / self.batch_size))
        data = data.reset_index(drop=True)
        data['batch'] = (data.index // self.batch_size) + 1

        return data
    
    def load_for_conditional_generation(self, model_name, explcit_prompting, **kwargs):
        """Load input data for LLM generation - to evaluate uphold_reasons_internal/uphold_reasons_external reliance"""

        # the reason why the data is stored in folders is to accomodate batched loading in the future
        directory_path = Path(PARSE_OUTPUT_PATH+'/'+model_name.split('/')[1]+'/'+self.data_name+'/'+"justify"+explcit_prompting)
        directory_path.mkdir(parents=True, exist_ok=True)
        file_path = directory_path / ("extracted_info.pkl")
        with file_path.open("rb") as f:
            extracted_inputs_reasons = pickle.load(f)
        return extracted_inputs_reasons
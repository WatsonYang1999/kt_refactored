import argparse
import json

class KTConfig:
    def __init__(self, json_file):
        self.args = self._load_from_json(json_file)
        self.model_config = self._extract_config_section("model_config")
        self.train_config = self._extract_config_section("train_config")
        self.dataset_config = self._extract_config_section("dataset_config")

    def _load_from_json(self, json_file):
        with open(json_file, 'r') as f:
            config = json.load(f)
            # Merge all configurations
            all_configs = {}
            for key, section in config.items():
                if isinstance(section, dict):
                    all_configs.update(section)
            return argparse.Namespace(**all_configs)

    def _extract_config_section(self, section_name):
        """Dynamically extracts a specific configuration section."""
        with open(json_file, 'r') as f:
            config = json.load(f)
            return {k: v for k, v in vars(self.args).items() if k in config.get(section_name, {})}

        
    def get_model_config(self):
        return self.model_config
    
    def get_dataset_config(self):
        return self.dataset_config
    
    def get_train_config(self):
        return self.train_config
    

    def _parse_args(self):
        def str_to_bool(s):
            return s.lower() == 'true'

        parser = argparse.ArgumentParser()

        # Model configuration
        model_group = parser.add_argument_group('model_config')
        model_group.add_argument('--model', type=str, default='DKVMN_RE',
                                 help='Model type to use, support GKT, SAKT, QGKT and DKT.')
        model_group.add_argument('--hidden_dim', type=int, default=50)
        model_group.add_argument('--embed_dim', type=int, default=128)
        model_group.add_argument('--output_dim', type=int, default=100)
        model_group.add_argument('--dropout', type=float, default=0.2)
        model_group.add_argument('--memory_size', type=int, default=20)
        model_group.add_argument('--n_heads', type=int, default=4)
        model_group.add_argument('--graph_type', type=str, default='PAM')
        model_group.add_argument('--edge_types', type=int, default=2)

        # Training configuration
        train_group = parser.add_argument_group('train_config')
        train_group.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
        train_group.add_argument('--current_epoch', type=int, default=0)
        train_group.add_argument('--n_epochs', type=int, default=20, help='Total Epochs.')
        train_group.add_argument('--batch_size', type=int, default=256)
        train_group.add_argument('--max_seq_len', type=int, default=200)
        train_group.add_argument('--shuffle', type=str_to_bool, default='False')
        train_group.add_argument('--cuda', type=str_to_bool, default='True')
        train_group.add_argument('--data_augment', type=str_to_bool, default='False')
        train_group.add_argument('--pretrain', type=str, default='load')
        train_group.add_argument('--pretrain_embed_file', type=str, default='', help='path of the pretrain weight file')
        train_group.add_argument('--log_file', type=str, default='', help='path of the logging file')

        # Dataset configuration
        dataset_group = parser.add_argument_group('dataset_config')
        dataset_group.add_argument('--dataset', type=str, default='ednet-re', help='Dataset You Wish To Load')
        dataset_group.add_argument('--checkpoint_dir', type=str, default=None, help='Model Parameters Directory')
        dataset_group.add_argument('--train_from_scratch', type=str_to_bool, default=False,
                                   help='If you need to retrain the model from scratch')
        dataset_group.add_argument('--eval', type=str_to_bool, default='False',
                                   help='Evaluate model to find some interesting insights')
        dataset_group.add_argument('--custom_data', type=str_to_bool, default='False',
                                   help='Use your own custom data')
        dataset_group.add_argument('--skill_level_eval', type=str_to_bool, default='False',
                                   help='Evaluate the model on skill level and try to address the label leakage issue')
        dataset_group.add_argument('--s_num', type=int, default=-1)
        dataset_group.add_argument('--q_num', type=int, default=-1)

        return parser.parse_args()

    def _load_from_json(self, json_file):
        with open(json_file, 'r') as f:
            config = json.load(f)
            # Convert the config dict into an argparse.Namespace
            model_config = config.get('model_config', {})
            train_config = config.get('train_config', {})
            dataset_config = config.get('dataset_config', {})

            # Merge all configurations
            all_configs = {**model_config, **train_config, **dataset_config}
            return argparse.Namespace(**all_configs)
            
            
    def update_qs(self, q_num, s_num, qs_matrix):
        self.model_config['q_num'] = q_num
        self.model_config['s_num'] = s_num
        self.model_config['qs_matrix'] = qs_matrix
# Example usage
# config_from_args = KTConfig(from_args=True)
# config_from_json = KTConfig(from_args=False, json_file='config.json')

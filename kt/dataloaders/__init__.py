def get_dataset_loader(dataset_config):
    print(dataset_config)
    if dataset_config['dataset'] == 'assist09_q':
        from .assist09_q_loader import Assistment09Loader
        return Assistment09Loader()


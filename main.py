from kt.config import KTConfig
from kt.dataloaders import get_dataset_loader
from kt.model import get_model
from kt.train import TrainManager

config = KTConfig(from_args=False, json_file='config/config.json')

loaders = get_dataset_loader(config.get_dataset_config())

config.update_qs()
print("Done Loading Loaders")
model = get_model(config.get_model_config())
print("Done Loading Models")
train_manager = TrainManager(config.get_train_config())
print("Done Loading TrainManager")
print(loaders)
print(model)

train_manager.train()
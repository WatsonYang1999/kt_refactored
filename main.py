from kt.config import KTConfig
from kt.dataloaders import get_dataset_loader
from kt.model import get_model
from kt.train import TrainManager
from kt.logger import KTLogger

logger = KTLogger().get_logger()


config = KTConfig(from_args=False, json_file='config/config.json')

loader = get_dataset_loader(config.get_dataset_config())

config.update_qs(loader.get_q_num(), loader.get_s_num(),loader.get_qs_matrix())
logger.info("Done Loading Loaders")
model = get_model(config.get_model_config())
logger.info("Done Loading Models")
train_manager = TrainManager(model, loader, config.get_train_config())
logger.info("Done Loading TrainManager")
logger.info(loader)
logger.info(model)

train_manager.train()
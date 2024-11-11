from abc import abstractmethod, abcfrom sklearn.model_selection import train_test_split
from kt.logger import KTLogger
logger = KTLogger().get_logger()

class BaseLoader:
    '''
        子类Loader继承BaseLoader，实现不同的初始化方法
    '''

    def __init__(self):
        self.user_sequences = None
    '''
        子类Loader继承BaseLoader,支持从不同数据集的文件，不同类型txt,csv等文件中加载数据
    '''

    def _load(self):
        parse_args
    def get_loader(self, ratios=(0.7, 0.2, 0.1)):
        seq_list = self._split_data(ratios)
        return {
            'train': seq_list[0],
            'test': seq_list[1],
            'valid': seq_list[2],
        }

    def _split_data(self, ratios=(0.7, 0.2, 0.1)):
        """
        Split user sequences into multiple groups based on specified ratios.

        :param self.user_sequences: Dictionary of user sequences {user_id: sequence}
        :param ratios: Tuple indicating the split ratios (e.g., (0.7, 0.2, 0.1) for train, val, test)
        :return: List of dictionaries containing split user sequences
        """
        #assert sum(ratios) < 1 + 1e-7 and sum(ratios) > 1 - 1e-7, "Ratios must sum to 1."
        #assert len(ratios) == 3, "Three ratios must be provided for train, validation, and test sets."
        logger.info(f"ratios: {ratios}")
        user_ids = list(self.user_sequences.keys())
        train_ratio, val_ratio, test_ratio = ratios
        # Split users for train set
        train_ids, temp_ids = train_test_split(user_ids, train_size=train_ratio, random_state=42)

        # Adjust validation and test ratios for the remaining data
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(temp_ids, train_size=val_ratio_adjusted, random_state=42)

        # Create split data based on user_ids
        split_data = [
            {uid: self.user_sequences[uid] for uid in train_ids},
            {uid: self.user_sequences[uid] for uid in val_ids},
            {uid: self.user_sequences[uid] for uid in test_ids}
        ]
        return split_data
import unittest
from unittest.mock import patch, mock_open
import pandas as pandas
from io import StringIO
from kt.dataloaders.assist09_q_loader import Assistment09Loader

import logging
logger = logging.getLogger(__file__)

class TestAssistment09Loader(unittest.TestCase):
    def setUp(self):
        self.fields_dtype = {
            'order_id': int,
            'user_id': int,
            'question_id': int,
            'correct': int,
            'skill_id': 'Int64',  # Using nullable integer for optional fields
            'skill_name': str,
            'time_elapsed': float
        }

    def test_load_with_extra_fields(self):
        assist_loader = Assistment09Loader()
        loaders = assist_loader.get_loader()

        logger.info(loaders['train'])
        logger.info(loaders['test'])
        logger.info(loaders['valid'])
        for dataset_type in {'train', 'test', 'valid'}:
            for k in self.fields_dtype:
                self.assertGreater(len(loaders[dataset_type][k]), 0)

        self.assertEqual(len(loaders), 3)
        
        
if __name__ == "__main__":
    unittest.main()

import numpy as np
import pandas as pd
from kt.dataloaders.base_loader import BaseLoader
from kt.logger import KTLogger
logger = KTLogger().get_logger()


class Assistment09Loader(BaseLoader):
    """
    Loader for the Assistment 2009 dataset with optional extra field loading.
    """

    def __init__(self, file_path='./dataset/assist2009/skill_builder_data_corrected.csv', load_extra_fields=True):
        # Define standard and extra fields with default data types
        fields = ['order_id', 'user_id', 'problem_id', 'correct']
        extra_fields = ['skill_id', 'skill_name', 'overlap_time']

        dtypes = {
            'order_id': int,
            'user_id': int,
            'problem_id': int,
            'correct': int,
            'skill_id': 'Int64',  # Using nullable integer for optional fields
            'skill_name': str,
            'overlap_time': float        }

        # Add extra fields if load_extra_fields is True
        if load_extra_fields:
            fields += extra_fields
        # Initialize the parent class (BaseLoader)
        super().__init__()

        # Store the fields and data types in instance variables
        self.file_path = file_path 
        self.fields = fields 
        self.dtypes = dtypes       
        self.load_extra_fields = load_extra_fields
        self.q_key = 'problem_id'
        self.s_key = 'skill_id'
        
        self.q_num = None
        self.s_num = None
        self.qs_matrix = None
        
        self.user_sequences = self._load()

    def get_q_num(self):
        if self.q_num is None:
            raise NotImplementedError('q_num not init yet')
        return self.q_num
    def get_s_num(self):
        if self.s_num is None:
            raise NotImplementedError('s_num not init yet')
        return self.s_num
    
    def get_qs_matrix(self):
        if self.qs_matrix is None:
            raise NotImplementedError('qs_matrix not init yet')
    
        return self.qs_matrix
    
    def get_qn_sn_qs_matrix(self):
        return (self.get_q_num(), self.get_s_num(), self.get_qs_matrix()) 

    def _load(self):
        """Load and clean the Assistment 2009 dataset, returning user sequences."""
        try:
            # Load specified fields with predefined data types
            df = pd.read_csv(self.file_path, usecols=self.fields, dtype={k: self.dtypes[k] for k in self.fields}, encoding='utf-8')

            # ToDo: 
            df = df.head(50)
            # df = df.sort_values('order_id', ascending=True).dropna().drop_duplicates(subset='order_id')
            df = df.sort_values('order_id', ascending=True).dropna()

            # reset q/s index from 1
            df[self.q_key], _ = pd.factorize(df[self.q_key]) 
            df[self.s_key], _ = pd.factorize(df[self.s_key])
            df[self.q_key] += 1
            df[self.s_key] += 1

            self._calculate_qs_relationship(df, q_key=self.q_key, s_key=self.s_key)

            logger.info(f'Original Data Count {df.shape[0]}')
            df = df.drop_duplicates(subset='order_id')
            logger.info(f'Drop Duplicate Data Count {df.shape[0]}')
            logger.info(df.columns)
            # Handle any additional processing needed for extra fields
            if self.s_key in df.columns:
                df[self.s_key] = df[self.s_key].fillna(-1).astype(int)  # Fill NaNs for optional fields
            if 'overlap_time' in df.columns:
                df['overlap_time'] = df['overlap_time'].fillna(0.0)
        except Exception as e:
            print(f"Error loading data: {e}")
            exit(-1)

        # Group by user and generate multi-dimensional sequences
        df_user_groups = df.groupby('user_id')
        user_sequences = {}

        for uid, df_user in df_user_groups:
            sequence = []
            for _, row in df_user.iterrows():
                # Create a dictionary of all relevant fields for each interaction
                interaction = {field: row[field] for field in self.fields if field in df_user.columns}
                sequence.append(interaction)
                # logger.info(interaction)
            user_sequences[uid] = sequence
        return user_sequences    
    
    def _calculate_qs_relationship(self, df: pd.DataFrame, q_key: str, s_key: str):        
        qs_counts = df.groupby([q_key, s_key]).size().reset_index(name='count')
        sq_counts = df.groupby([s_key, q_key]).size().reset_index(name='count')
        
        logger.info(qs_counts)
        logger.info(sq_counts)

        # q_to_s_count = df.groupby(q_key)[s_key].nunique()

        # # 统计每个 B 对应的 A 的数量
        # s_to_q_count = df.groupby(s_key)[q_key].nunique()

        # 计算 A 平均对应几个 B
        # avg_q_to_s = q_to_s_count.mean()

        # # 计算 B 平均对应几个 A
        # avg_s_to_q = s_to_q_count.mean()

        # # 输出结果
        # print(f"每个 q 平均对应 {avg_q_to_s:.2f} 个 s")
        # print(f"每个 s 平均对应 {avg_s_to_q:.2f} 个 q")

        q_unique = df[q_key].unique().tolist()
        s_unique = df[s_key].unique().tolist()
        
        self.q_num = q_unique
        self.s_num = s_unique
        
        self.qs_matrix = np.zeros((len(q_unique) + 1, len(s_unique) + 1), dtype=int)

        for _, row in df.iterrows():
            self.qs_matrix[row[self.q_key], row[self.s_key]] = 1
        # print(f"每个 q 平均对应 {np.sum(self.qs_matrix) / (self.qs_matrix.shape[0] - 1):.2f} 个 s")
        # print(f"每个 s 平均对应 {np.sum(self.qs_matrix) / (self.qs_matrix.shape[1] - 1):.2f} 个 q")

if __name__ == '__main__':
    loader = Assistment09Loader()

    logger.info(loader.get_loader())

import torch
from torch.utils.data import Dataset, DataLoader


class UserInteractionDataset(Dataset):
    def __init__(self, user_data, max_length=100, padding_value=-1):
        self.user_data = user_data  # Dictionary containing user interaction records
        self.max_length = max_length
        self.padding_value = padding_value

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        user_id = list(self.user_data.keys())[idx]
        interactions = self.user_data[user_id]
        # Extract individual features and pad/truncate each feature tensor
        question_ids = [interaction['question_id']
                        for interaction in interactions]
        correct_answers = [interaction['correct']
                           for interaction in interactions]
        skill_ids = [interaction['skill_id'] for interaction in interactions]
        overlap_times = [interaction['overlap_time']
                         for interaction in interactions]

        # Pad or truncate to max_length
        question_ids_tensor = self.pad_or_truncate(
            question_ids, dtype=torch.long)
        correct_answers_tensor = self.pad_or_truncate(
            correct_answers, dtype=torch.long)
        skill_ids_tensor = self.pad_or_truncate(skill_ids, dtype=torch.long)
        overlap_times_tensor = self.pad_or_truncate(
            overlap_times, dtype=torch.float)
        # Create a dictionary for each tensor slice
        data_tensor = {
            'question_id': question_ids_tensor,
            'correct': correct_answers_tensor,
            'skill_id': skill_ids_tensor,
            'overlap_time': overlap_times_tensor,
        }

        return data_tensor

    def pad_or_truncate(self, data, dtype=torch.float):
        # Pad with the padding value or truncate to max_length
        padded_data = data[:self.max_length] + \
            [self.padding_value] * max(0, self.max_length - len(data))
        return torch.tensor(padded_data, dtype=dtype)


# Sample usage:
user_data = {
    1: [
        {'order_id': 1, 'user_id': 1, 'question_id': 13, 'correct': 1,
            'skill_id': 23, 'skill_name': 'Box and Whisker', 'overlap_time': 1.0},
        {'order_id': 2, 'user_id': 1, 'question_id': 14, 'correct': 0,
            'skill_id': 24, 'skill_name': 'Box and Whisker', 'overlap_time': 2.0}
    ],
    2: [
        {'order_id': 3, 'user_id': 1, 'question_id': 13, 'correct': 1,
            'skill_id': 25, 'skill_name': 'Box and Whisker', 'overlap_time': 1.0},
        {'order_id': 4, 'user_id': 1, 'question_id': 14, 'correct': 0,
            'skill_id': 26, 'skill_name': 'Box and Whisker', 'overlap_time': 2.0}
    ],
    3: [
        {'order_id': 5, 'user_id': 1, 'question_id': 13, 'correct': 1,
            'skill_id': 27, 'skill_name': 'Box and Whisker', 'overlap_time': 1.0},
        {'order_id': 6, 'user_id': 1, 'question_id': 14, 'correct': 0,
            'skill_id': 28, 'skill_name': 'Box and Whisker', 'overlap_time': 2.0}
    ],
    4: [
        {'order_id': 7, 'user_id': 1, 'question_id': 13, 'correct': 1,
            'skill_id': 29, 'skill_name': 'Box and Whisker', 'overlap_time': 1.0},
        {'order_id': 8, 'user_id': 1, 'question_id': 14, 'correct': 0,
            'skill_id': 30, 'skill_name': 'Box and Whisker', 'overlap_time': 2.0}
    ],
}

dataset = UserInteractionDataset(user_data, max_length=5, padding_value=-1)

# DataLoader for batching
train_loader = DataLoader(dataset, batch_size=2, shuffle=False)

# Retrieve the first batch and print
for batch in train_loader:
    print(batch)
    break

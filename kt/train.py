import torch
from torch.utils.data import DataLoader
import os
from kt.interaction_dataset import UserInteractionDataset


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}
class TrainManager:
    def __init__(self, model, dataset_loader, config):
        """
        初始化 TrainManager 实例。
        
        参数:
        - model_class: 要加载的模型类。
        - dataset_class: 要加载的数据集类。
        - config: 包含训练配置的字典。
        """
        print(config)
        self.config = config
        self.device = torch.device("cuda" if config['cuda'] and torch.cuda.is_available() else "cpu")
        
        # 初始化模型并加载到设备
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.criterion = self._load_loss_function(config['loss'])
        
        # 加载数据集并创建 DataLoader
        print('---------------------------------------')
        print(dataset_loader.get_loader()['train'])
        self.train_loader = DataLoader(
            UserInteractionDataset(dataset_loader.get_loader()['train'],max_length=config['max_seq_len']),
            batch_size=config['batch_size'],
            shuffle=config['shuffle']
        )
        print('---------------------------------------')

        
        self.valid_loader = DataLoader(
            dataset_loader.get_loader()['test'],
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # 日志文件设置
        self.log_file = open(config['log_file'], 'a') if config['log_file'] else None
        self.start_epoch = config['current_epoch']
        self.n_epochs = config['n_epochs']
        
        # 加载预训练权重（如果需要）
        if config['pretrain'] == 'load' and config['pretrain_embed_file']:
            self._load_pretrained_weights(config['pretrain_embed_file'])
        
        # 检查点目录
        # self.checkpoint_dir = config['checkpoint_dir']
        # if self.checkpoint_dir and not os.path.exists(self.checkpoint_dir):
        #     os.makedirs(self.checkpoint_dir)
            
    def _load_loss_function(self, loss_type):
        from kt.model.loss import KTLoss
        return KTLoss()

    def _load_pretrained_weights(self, weight_file):
        if os.path.exists(weight_file):
            self.model.load_state_dict(torch.load(weight_file))
            print("Pretrained weights loaded.")
        else:
            print("Warning: Pretrained weight file not found.")

    def _save_checkpoint(self, epoch):
        if self.checkpoint_dir:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    def _log(self, message):
        print(message)
        if self.log_file:
            self.log_file.write(message + "\n")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            print(batch)
            batch = move_batch_to_device(batch,device=self.config['device'])
            
            # Forward pass
            outputs = self.model(batch)
            loss = self.criterion(outputs, labels)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        self._log(f"Epoch [{epoch}/{self.n_epochs}], Train Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.valid_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.valid_loader)
        self._log(f"Epoch [{epoch}/{self.n_epochs}], Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        for epoch in range(self.start_epoch, self.n_epochs):
            # Training step
            self.train_one_epoch(epoch)
            
            # Validation step
            if epoch % 1 == 0 or epoch == self.n_epochs - 1:  # 可以调整验证的频率
                self.validate(epoch)
            
            # 保存检查点
            if self.checkpoint_dir and (epoch % 5 == 0 or epoch == self.n_epochs - 1):
                self._save_checkpoint(epoch)
        
        # 关闭日志文件
        if self.log_file:
            self.log_file.close()
        print("Training complete.")

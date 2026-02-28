# light_logger.py
import os
import logging
from datetime import datetime

class LightLogger:
    def __init__(self, model_name, dataset_name, train_ratio, log_dir='./logs'):
        """完整的轻量级日志记录器（含训练时间记录）"""
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio
        self.log_dir = log_dir
        self.start_time = datetime.now()  # 记录开始时间
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 生成带时间戳的日志文件名
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            self.log_dir,
            f"{model_name}_{dataset_name}_{train_ratio}_{timestamp}.log"
        )
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[logging.FileHandler(self.log_file)]
        )
        self.logger = logging.getLogger()
        
        # 记录实验开始信息
        self.logger.info(f"===== 实验开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} =====")
        self.logger.info(f"模型名称: {model_name}")
        self.logger.info(f"数据集: {dataset_name}")
        self.logger.info(f"训练比例: {train_ratio}\n")
    
    def log_training_time(self):
        """记录训练总耗时"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info("\n===== 训练时间统计 =====")
        self.logger.info(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"总耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    
    def log_config(self, config, model=None):
        """完整记录实验配置和模型参数"""
        self.logger.info("===== 实验配置 =====")
        
        # 记录训练配置
        config_params = ['seed', 'lr', 'batch_size', 'epochs', 'weight_decay']
        for param in config_params:
            if hasattr(config, param):
                self.logger.info(f"{param}: {getattr(config, param)}")
        
        # 记录模型参数
        if model:
            self.logger.info("\n===== 模型参数 =====")
            if self.model_name == 'LTP':
                params = ['latent_dim', 'patch_len', 'stride', 'num_layers', 'nhead', 'period']
            elif self.model_name == 'NLTP':
                params = ['latent_dim', 'period', 'num_scales']
            
            for p in params:
                if hasattr(model, p):
                    self.logger.info(f"{p}: {getattr(model, p)}")
        self.logger.info("")
    
    def log_metrics(self, metrics_dict, phase="结果"):
        """完整记录评估指标"""
        self.logger.info(f"===== {phase} =====")
        for metric, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{metric}: {value:.6f}")
            else:
                self.logger.info(f"{metric}: {value}")
        self.logger.info("")
    
    def log_message(self, message):
        """记录自定义消息"""
        self.logger.info(message)
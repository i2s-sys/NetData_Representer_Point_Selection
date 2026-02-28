import configparser
import numpy as np
import torch
import math
import json
import os
from utils import idx2seq, tensor2tuple, tuple2tensor


class Config(object):
    def __init__(self, config_file, dataset_name=None):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % config_file)

        # Hyper-parameter
        self.epochs = conf.getint("Model_Setup", "epochs")
        self.lr = conf.getfloat("Model_Setup", "lr")
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.batch_size = conf.getint("Model_Setup", "bs")
        self.batch_size_eval = conf.getint("Model_Setup", "bs_eval")
        self.seed = conf.getint("Model_Setup", "seed")

        # Dataset
        self.num_dim = np.array(json.loads(conf.get("Data_Setting", "ndim")))
        self.num_emb = conf.getint("Data_Setting", "nemb")
        self.num_embs = np.array(json.loads(conf.get("Data_Setting", "nembs")))
        self.period = conf.getint("Data_Setting", "period")
        self.channels = conf.getint("Data_Setting", "channels")

        self.dataset_name = dataset_name if dataset_name is not None else conf.get("Data_Setting", "dataset_name", fallback="WSDreamTP")
        self.data_dir = conf.get("Data_Setting", "data_dir", fallback="./data/")
        self.data_path = conf.get("Data_Setting", "data_path")
        self.location_path = conf.get("Data_Setting", "location_path")

        self.num_batch = None
        self.num_batch_vali = None
        self.num_batch_test = None
        self.num_train = None
        self.max_value = None

    def Sampling(self, sample_ratio, mm):
        """
        加载采样数据，并支持按时间过滤不同阶段的数据点（用于评估时间点缺失补全）
        """
        trainset_path = f"{self.data_dir}{self.dataset_name}/trainset_{sample_ratio}.npy"
        validset_path = f"{self.data_dir}{self.dataset_name}/validset_{sample_ratio}.npy"
        testset_path = f"{self.data_dir}{self.dataset_name}/testset_{sample_ratio}.npy"

        train_data = np.load(trainset_path)
        valid_data = np.load(validset_path)
        test_data = np.load(testset_path)
        # 归一化
        raw_train = train_data[:, 3]
        raw_valid = valid_data[:, 3]
        raw_test = test_data[:, 3]

        max_val = 1

        tr_vals = torch.from_numpy(raw_train / max_val).float().cuda()
        va_vals = torch.from_numpy(raw_valid / max_val).float().cuda()
        te_vals = torch.from_numpy(raw_test / max_val).float().cuda()

        self.max_val = max_val

        tr_idxs = torch.from_numpy(train_data[:, :3].astype(int)).cuda().long()
        va_idxs = torch.from_numpy(valid_data[:, :3].astype(int)).cuda().long()
        te_idxs = torch.from_numpy(test_data[:, :3].astype(int)).cuda().long()

        self.num_batch = int(math.ceil(float(len(train_data)) / float(self.batch_size)))
        self.num_batch_vali = int(math.ceil(float(len(valid_data)) / float(self.batch_size_eval)))
        self.num_batch_test = int(math.ceil(float(len(test_data)) / float(self.batch_size_eval)))
        self.num_train = len(train_data)

        if mm in [1]:
            all_data = np.concatenate((train_data, valid_data, test_data))
            sequence = idx2seq(all_data[:, 2].astype(int), self.period)
            tr_seq = torch.from_numpy(sequence[:len(train_data)]).cuda().long()
            va_seq = torch.from_numpy(sequence[len(train_data):len(train_data) + len(valid_data)]).cuda().long()
            te_seq = torch.from_numpy(sequence[len(train_data) + len(valid_data):]).cuda().long()
        else:
            tr_seq = None
            va_seq = None
            te_seq = None

        return tr_idxs, tr_vals, va_idxs, va_vals, te_idxs, te_vals, tr_seq, va_seq, te_seq

import numpy as np
import torch
import csv

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        # if self.verbose:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)     # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


def evaluate_model(model, idxs, seq, vals, batch_size, criterion=None, device='cuda', missing_mask=None, return_preds=False):
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0 if criterion else None
    num_batches = (len(idxs) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for n in range(num_batches):
            start = n * batch_size
            end = (n + 1) * batch_size
            batch_idx = idxs[start:end]
            i = batch_idx[:, 0].to(device)
            j = batch_idx[:, 1].to(device)
            ks = seq[start:end].to(device)
            batch_vals = vals[start:end].to(device)

            if missing_mask is not None:
                batch_missing_mask = missing_mask[start:end].to(device)
                batch_pred = model(i, j, ks, batch_missing_mask)
            else:
                batch_pred = model(i, j, ks)

            preds.append(batch_pred.cpu().numpy())
            targets.append(batch_vals.cpu().numpy())

            if criterion is not None:
                total_loss += criterion(batch_pred, batch_vals).item()

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    nmae, nrmse = accuracy(preds, targets)

    if return_preds:
        if criterion is not None:
            return preds, (nmae, nrmse, total_loss / num_batches)
        return preds, (nmae, nrmse)
    else:
        if criterion is not None:
            return nmae, nrmse, total_loss / num_batches
        return nmae, nrmse


def accuracy(pred, real):
    """
    计算NMAE、NMSE和NRMSE，并限制预测值范围

    参数:
    pred (numpy.ndarray): 预测值数组
    real (numpy.ndarray): 真实值数组

    返回:
    tuple: (NMAE, NMSE, NRMSE)
    """
    # 限制预测值范围
    pred[pred <= 0] = 1e-8 
    pred[pred > 1] = 1     

    nmae = np.sum(np.fabs(real - pred)) / np.sum(np.fabs(real))
    nmse = np.sum(np.square(real - pred)) / np.sum(np.square(real))
    nrmse = np.sqrt(nmse)
    
    return nmae, nrmse

# def accuracy(predict, data):
#     error = np.abs(predict - data)
#     NMAE = np.sum(error) / np.sum(np.abs(data))
#     NRMSE = np.sqrt(np.sum(error ** 2) / np.sum(data ** 2))
#     return NMAE, NRMSE


# def accuracy(predict, data, num, num_time):
#     mask = (data > 0).astype(float)
#     error = np.multiply(mask, np.abs(predict - data))
#     NMAE = np.sum(error) / np.sum(np.abs(data))
#     NRMSE = np.sqrt(np.sum(error ** 2) / np.sum(data ** 2))
#     KL = scipy.stats.entropy(data.reshape(-1), (mask * predict).reshape(-1))
#     # NDCG = ndcg_score(data.reshape(num_time, -1), (mask * predict).reshape(num_time, -1), k=10)
#     NDCG = ndcg_score(data.transpose(2, 0, 1).reshape(num_time, -1), (mask * predict).transpose(2, 0, 1).reshape(num_time, -1), k=10)
#     MAE = np.sum(error) / num
#     RMSE = np.sqrt(np.sum(error ** 2) / num)
#     return NMAE, NRMSE, KL, NDCG, MAE, RMSE


def record(record_file, results):
    with open(record_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in results:
            writer.writerow(row)
        csvfile.close()


def tensor2tuple(tensor_):
    tuple_ = []
    [I, J, K] = list(tensor_.shape)
    for i in range(I):
        for j in range(J):
            for k in range(K):
                if tensor_[i, j, k] != 0:
                    tuple_ += [np.asarray([i, j, k, tensor_[i, j, k]])]
    return np.asarray(tuple_)

def tuple2tensor(indices, values):
    I = np.max(indices[:, 0])+1
    J = np.max(indices[:, 1])+1
    K = np.max(indices[:, 2])+1
    tensor_ = np.zeros((I, J, K))
    for n in range(len(values)):
        [i, j, k] = indices[n].tolist()
        tensor_[i, j, k] = values[n]
    return tensor_


def idx2seq(index, period=5):
    seq = []
    # for k in index:
    #     if k < period:
    #         tmp_seq = [0] * (period - k) + list(range(k))
    #     else:
    #         tmp_seq = list(range(k - period, k))
    for k in index:
        tmp_seq = list(range(k - period, k))
        seq += [tmp_seq]
    seq = np.asarray(seq)
    seq = np.clip(seq, 0, None)
    return seq


# def loc2mat():
#     data = np.load('./data/Harvard72Store.npy')
#     data = np.round(data, 
#     data = np.round(data, 4)
#     data[64, data[64] > 1e3] = 0        # 4e3
#     np.save('../tucker_compress/data/Harvard72.npy', data)
#
#     data = tensor2tuple(data)
#     loc = np.random.permutation(len(data))
#     matrix = tuple2tensor(data[:, 0:3].astype(int), loc+1)
#     np.save('../tucker_compress/data/Harvard72_loc.npy', matrix)

def create_missing_data(seq, missing_rate, pattern='random'):
    """
    创建缺失数据
    
    参数:
        seq: 时间序列 [N, period]
        missing_rate: 缺失率
        pattern: 缺失模式 ('random', 'consecutive', 'periodic')
    """
    seq_missing = seq.clone()
    N, period = seq.shape
    
    for i in range(N):
        num_missing = max(1, int(period * missing_rate))  # 确保至少有1个缺失
        
        if pattern == 'random':
            # 随机缺失
            missing_indices = np.random.choice(period, min(num_missing, period), replace=False)
        elif pattern == 'consecutive':
            # 连续缺失
            if num_missing >= period:
                missing_indices = np.arange(period)
            else:
                start = np.random.randint(0, period - num_missing + 1)
                missing_indices = np.arange(start, start + num_missing)
        elif pattern == 'periodic':
            # 周期性缺失（每隔一定间隔缺失）
            step = max(1, int(1 / missing_rate))
            missing_indices = np.arange(0, period, step)[:num_missing]
        
        # 标记缺失位置为-1
        seq_missing[i, missing_indices] = -1
    
    return seq_missing

def build_missing_mask(seq, missing_value=-1):
    """
    构建缺失数据掩码
    
    参数:
        seq: 时间序列 [N, period]
        missing_value: 缺失值标记
    
    返回:
        mask: 缺失掩码，True表示缺失
    """
    return seq == missing_value

def impute_missing_data(seq_missing, method='linear'):
    """
    补全缺失数据
    
    参数:
        seq_missing: 带缺失标记的序列 [N, period]
        method: 补全方法 ('linear', 'forward', 'backward', 'mean')
    """
    seq_imputed = seq_missing.clone().to(torch.float32)
    N, period = seq_missing.shape
    
    for i in range(N):
        missing_mask = seq_missing[i] == -1
        if missing_mask.sum() == 0:
            continue
            
        if method == 'linear':
            # 线性插值
            valid_indices = torch.where(~missing_mask)[0]
            if len(valid_indices) >= 2:
                missing_indices = torch.where(missing_mask)[0]
                interpolated_values = torch.tensor(
                    np.interp(missing_indices.cpu().numpy(),
                             valid_indices.cpu().numpy(),
                             seq_missing[i, valid_indices].cpu().numpy()),
                    device=seq_missing.device,
                    dtype=torch.float32
                )
                seq_imputed[i, missing_mask] = interpolated_values
        elif method == 'forward':
            # 前向填充
            last_valid = None
            for j in range(period):
                if not missing_mask[j]:
                    last_valid = seq_missing[i, j]
                elif last_valid is not None:
                    seq_imputed[i, j] = last_valid
                    
        elif method == 'backward':
            # 后向填充
            next_valid = None
            for j in range(period-1, -1, -1):
                if not missing_mask[j]:
                    next_valid = seq_missing[i, j]
                elif next_valid is not None:
                    seq_imputed[i, j] = next_valid
                    
        elif method == 'mean':
            # 均值填充
            valid_values = seq_missing[i, ~missing_mask]
            if len(valid_values) > 0:
                seq_imputed[i, missing_mask] = valid_values.mean()
    
    return seq_imputed
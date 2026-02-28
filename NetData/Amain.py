import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from Amodels import CP, CP_A, CP_B
from utils import accuracy, EarlyStopping
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--mm', type=int, default=0)
parser.add_argument('--dd', type=int, default=0)
parser.add_argument('--patience', type=int, default=50)
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpuid)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)

if __name__ == "__main__":
    mm = args.mm  # 0-CP, 1-CP+A, 2-CP+B
    Models = ['CP', 'CP_A', 'CP_B']
    dd = args.dd  # 0-Abilene, 1-Geant, 2-WSDreamTP
    Datasets = ['Abilene', 'Geant', 'WSDreamTP']
    config = Config(f'./data/{Datasets[dd]}.ini', dataset_name=Datasets[dd])

    train_ratio = 0.1
    checkpoint_dir = ('./results/checkpoint/{}_{}_{}.pt').format(Models[mm], Datasets[dd], train_ratio)

    print("Dataset - " + Datasets[dd])
    tr_idxs, tr_vals, va_idxs, va_vals, te_idxs, te_vals, tr_seq, va_seq, te_seq = config.Sampling(train_ratio, mm)

    print("Model - " + Models[mm])
    if mm == 0:
        model = CP(config.num_dim, config.num_emb)
    elif mm == 1:
        model = CP_A(config.num_dim, config.num_emb)
    elif mm == 2:
        model = CP_B(config.num_dim, config.num_emb)
    else:
        raise Exception("Unknown model")

    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.L1Loss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=checkpoint_dir)

    print('Training...')
    for epoch in range(config.epochs):
        model.train()
        random_perm_idx = np.random.permutation(int(config.num_train))
        train_loss = 0.0

        for n in range(config.num_batch):
            batch_set_idx = random_perm_idx[n * config.batch_size: (n + 1) * config.batch_size]
            i = tr_idxs[batch_set_idx][:, 0]
            j = tr_idxs[batch_set_idx][:, 1]
            k = tr_idxs[batch_set_idx][:, 2]
            val = tr_vals[batch_set_idx]
            ks = tr_seq[batch_set_idx] if mm in [1] else None

            optimizer.zero_grad()
            predict = model(i, j, k, ks)
            loss = criterion(predict, val)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= config.num_batch

        # === 每个 epoch 输出一次训练集和验证集性能指标 ===
        model.eval()
        with torch.no_grad():
            Estimated_train = []
            for n in range(config.num_batch):
                idxs_batch = tr_idxs[n * config.batch_size: (n + 1) * config.batch_size]
                i = idxs_batch[:, 0]
                j = idxs_batch[:, 1]
                k = idxs_batch[:, 2]
                ks = tr_seq[n * config.batch_size: (n + 1) * config.batch_size] if mm in [1, 3] else None
                predict = model(i, j, k, ks)
                Estimated_train += predict.cpu().numpy().tolist()
            Estimated_train = np.asarray(Estimated_train)
            train_nmae, train_nrmse = accuracy(Estimated_train, tr_vals.cpu().numpy())

            Estimated_valid = []
            valid_loss = 0
            for n in range(config.num_batch_vali):
                idxs_batch = va_idxs[n * config.batch_size_eval: (n + 1) * config.batch_size_eval]
                i = idxs_batch[:, 0]
                j = idxs_batch[:, 1]
                k = idxs_batch[:, 2]
                val = va_vals[n * config.batch_size_eval: (n + 1) * config.batch_size_eval]
                ks = va_seq[n * config.batch_size_eval: (n + 1) * config.batch_size_eval] if mm in [1, 3] else None

                predict = model(i, j, k, ks)
                Estimated_valid += predict.cpu().numpy().tolist()
                valid_loss += criterion(predict, val).item()

            valid_loss /= config.num_batch_vali
            Estimated_valid = np.asarray(Estimated_valid)
            valid_nmae, valid_nrmse = accuracy(Estimated_valid, va_vals.cpu().numpy())

            print(f"Epoch [{epoch + 1}/{config.epochs}]    train_loss: {train_loss:.6f}")
            print(f"Train NMAE: {train_nmae:.4f}, Train NRMSE: {train_nrmse:.4f}")
            print(f"Valid NMAE: {valid_nmae:.4f}, Valid NRMSE: {valid_nrmse:.4f}, Valid Loss: {valid_loss:.6f}")

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Testing...")
    model.load_state_dict(torch.load(checkpoint_dir))
    model.eval()

    with torch.no_grad():
        Estimated_test = []
        for n in range(config.num_batch_test):
            idxs_batch = te_idxs[n * config.batch_size_eval: (n + 1) * config.batch_size_eval]
            i = idxs_batch[:, 0]
            j = idxs_batch[:, 1]
            k = idxs_batch[:, 2]
            ks = te_seq[n * config.batch_size_eval: (n + 1) * config.batch_size_eval] if mm in [1, 3] else None
            predict = model(i, j, k, ks)
            Estimated_test += predict.cpu().numpy().tolist()

    Estimated_test = np.asarray(Estimated_test)
    test_nmae, test_nrmse = accuracy(Estimated_test, te_vals.cpu().numpy())

    print("Final Test NMAE: {:.4f}, Test NRMSE: {:.4f}".format(test_nmae, test_nrmse))

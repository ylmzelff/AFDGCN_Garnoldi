import torch
import warnings
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler
warnings.filterwarnings('ignore')

SLOTS_PER_DAY = 144  # 10-dakikalik araliklar, 24*6=144

def add_temporal_features(data, tod=False, dow=False):
    """
    Normalize edilmis (T, N, D) verisine sin/cos zamansal ozellikler ekler.
    tod=True: gunun saati (slot 0-143 -> sin/cos)  → D += 2
    dow=True: haftanin gunu (gun 0-6  -> sin/cos)  → D += 2
    Sin/cos encoding zaten [-1,1] araliginda, ek normalizasyon gerekmez.
    """
    T, N, _ = data.shape
    features = [data]

    if tod:
        slot_idx = np.arange(T) % SLOTS_PER_DAY
        sin_tod = np.sin(2 * np.pi * slot_idx / SLOTS_PER_DAY).astype(np.float32)
        cos_tod = np.cos(2 * np.pi * slot_idx / SLOTS_PER_DAY).astype(np.float32)
        features.append(np.tile(sin_tod[:, None, None], (1, N, 1)))
        features.append(np.tile(cos_tod[:, None, None], (1, N, 1)))

    if dow:
        day_idx = (np.arange(T) // SLOTS_PER_DAY) % 7
        sin_dow = np.sin(2 * np.pi * day_idx / 7).astype(np.float32)
        cos_dow = np.cos(2 * np.pi * day_idx / 7).astype(np.float32)
        features.append(np.tile(sin_dow[:, None, None], (1, N, 1)))
        features.append(np.tile(cos_dow[:, None, None], (1, N, 1)))

    result = np.concatenate(features, axis=-1)
    print(f'Temporal features added: tod={tod}, dow={dow} | shape {data.shape} -> {result.shape}')
    return result

def normalize_dataset(data, normalizer):
    if normalizer == 'max01':
        minimum = data.min()
        maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization', minimum, maximum)
    elif normalizer == 'max11':
        minimum = data.min()
        maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization', minimum, maximum)
    elif normalizer == 'std':
        mean = data.mean()
        std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization', mean, std)
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60) / interval)
    test_days = int(test_days)
    val_days  = int(val_days)
    test_data = data[-T * test_days:]
    val_data = data[-T * (test_days + val_days): -T * test_days]
    train_data = data[:-T * (test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio+val_ratio)) : -int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=False):
    # 1.加载数据集
    print(f"DEBUG: args.dataset = '{args.dataset}'")
    print(f"DEBUG: args.num_nodes = {args.num_nodes}")
    data = load_st_dataset(args.dataset)        # B, N, D
    # 2.数据归일화처리 (sadece flow feature'ini normalize et)
    data, scaler = normalize_dataset(data, normalizer)
    # 3. Zamansal ozellikler (normalize edilmis akis + sin/cos zamansal kodlama)
    if tod or dow:
        data = add_temporal_features(data, tod=tod, dow=dow)
    # 4.数据集划分(训练集、验证集、测试集)
    if args.test_ratio > 1:
        # interval=10: verimiz 10 dakikalik aralikli (144 slot/gun)
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio, interval=10)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    # 5.滑动窗口采样
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    # 5.生成数据迭代器dataloader，并对训练集样本进行打乱
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    
    return train_dataloader, val_dataloader, test_dataloader, scaler


if __name__ == '__main__':
    import argparse
    DATASET = 'PEMS08'
    NODE_NUM = 170
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=False)
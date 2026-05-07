import os
import numpy as np

def load_st_dataset(dataset):
    # output B, N, D
    print(f"DEBUG: load_st_dataset called with dataset = '{dataset}'")
    if dataset == 'PEMS04':
        # Try different PEMS04 data files
        npz_path = '/content/AFDGCN_Garnoldi/data/PEMS04/PEMS04.npz'
        morning_path = '/content/AFDGCN_Garnoldi/data/PEMS04/morning_only.npz'
        
        if os.path.exists(npz_path):
            print(f"Loading PEMS04 from {npz_path}")
            data = np.load(npz_path)['data'][:, :, :1]
        elif os.path.exists(morning_path):
            print(f"Loading PEMS04 from {morning_path}")
            data = np.load(morning_path)['data'][:, :, :1]
        else:
            print(f"WARNING: PEMS04 data not found! Creating synthetic data (307 nodes)")
            # Create synthetic PEMS04-compatible data
            data = np.random.rand(16992, 307, 1) * 50 + 10  # ~2 months, 307 nodes
    elif dataset == 'PEMS08':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, :1]  # only the first dimension, traffic flow data
    elif dataset == 'PEMS03':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, :1]  # only the first dimension, traffic flow data
    elif dataset == 'PEMS07':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, :1]  # only the first dimension, traffic flow data
    elif dataset == 'Kcetas':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/Kcetas/morning_only.npz')
        data = np.load(data_path)['data'][:, :, :1] 
    elif dataset == 'Konya':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/Konya/konya_kavşak.npz')
        data = np.load(data_path)['data'][:, :, :1]  # only the first dimension, traffic flow data
    elif dataset == 'Kayseri':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/Kayseri/kol_bazli_kis.npz')
        data = np.load(data_path)['data'][:, :, :1]  # only the first dimension, traffic flow data
    elif dataset == 'Kayseri_Serit':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/Kayseri/kol_bazli_kis.npz')
        data = np.load(data_path)['data'][:, :, :1]  # only the first dimension, traffic flow data
    elif dataset == 'Sivas':
        data_path = os.path.join('/content/AFDGCN_Garnoldi/data/Sivas/kol_bazli.npz')
        data = np.load(data_path)['data'][:, :, :1]  # only the first dimension, traffic flow data
    else:
        raise ValueError
    print('Load %s Dataset shaped: ' % dataset, data.shape)
    return data

# 测试内容
# data = load_st_dataset(dataset = 'PEMS08')
# print(data.shape)

import numpy as np
import pandas as pd

def npz_to_csv(npz_file_path, output_csv_path):
    """
    NPZ dosyasını CSV formatına dönüştürür
    """
    # NPZ dosyasını yükle
    data = np.load(npz_file_path)
    
    # NPZ dosyasındaki anahtarları görüntüle
    print("NPZ dosyasındaki anahtarlar:", list(data.keys()))
    
    # Ana veri arrayini al (genellikle 'data' anahtarında)
    if 'data' in data.keys():
        array_data = data['data']
    else:
        # İlk anahtarı al
        first_key = list(data.keys())[0]
        array_data = data[first_key]
    
    print(f"Veri boyutu: {array_data.shape}")
    
    # Veri formatını kontrol et
    if len(array_data.shape) == 3:  # (timesteps, nodes, features)
        timesteps, nodes, features = array_data.shape
        print(f"Format: {timesteps} timestep, {nodes} node, {features} feature")
        
        # CSV için veriyi düzenle
        rows = []
        for t in range(timesteps):
            for n in range(nodes):
                row = {
                    'timestep': t + 1,
                    'node': n,
                }
                # Her feature için sütun ekle
                for f in range(features):
                    row[f'feature_{f}'] = array_data[t, n, f]
                rows.append(row)
        
        # DataFrame oluştur ve CSV'ye kaydet
        df = pd.DataFrame(rows)
        df.to_csv(output_csv_path, index=False)
        print(f"CSV dosyası kaydedildi: {output_csv_path}")
        
    elif len(array_data.shape) == 2:  # (samples, features)
        samples, features = array_data.shape
        print(f"Format: {samples} sample, {features} feature")
        
        # DataFrame oluştur
        columns = [f'feature_{i}' for i in range(features)]
        df = pd.DataFrame(array_data, columns=columns)
        df.to_csv(output_csv_path, index=False)
        print(f"CSV dosyası kaydedildi: {output_csv_path}")
    
    else:
        print(f"Desteklenmeyen veri formatı: {array_data.shape}")
        return
    
    # İlk birkaç satırı göster
    print("\nİlk 5 satır:")
    print(df.head())

if __name__ == "__main__":
    # Dosya yollarını belirle
    npz_file = input("NPZ dosya yolu: ").strip()
    csv_file = input("Çıktı CSV dosya adı (örn: output.csv): ").strip()
    
    if not csv_file.endswith('.csv'):
        csv_file += '.csv'
    
    try:
        npz_to_csv(npz_file, csv_file)
    except Exception as e:
        print(f"Hata: {e}")
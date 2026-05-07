import pandas as pd
from datetime import datetime, timedelta

def convert_to_intersection_format(input_csv, output_excel):
    # CSV dosyasını oku
    df = pd.read_csv(input_csv)
    
    # Kavşak ve kol tanımları
    intersections = {
        'Gesi': {
            'locations': [0, 1, 2, 3], 
            'cols': ['A - SİVAS BULVARI-SİVAS YÖNÜ', 'B - GESİ CAD.', 'C - SİVAS BULV - ŞEHİR MERKEZİ', 'D - 381. SOKAK']
        },
        'Serkent': {
            'locations': [4, 5, 6, 7], 
            'cols': ['A - 822. SK', 'B - GESİ CAD. DOĞU', 'C - KOCASİNAN CAD.', 'D - GESİ CAD. BATI']
        },
        'Beyazşehir': {
            'locations': [8, 9, 10, 11], 
            'cols': ['A - OSMAN ÖZCAN CAD.', 'B - GESİ CAD DOĞU', 'C - MARKETLER ÇIKIŞ', 'D - GESİ CAD BATI']
        },
        'Toki': {
            'locations': [12, 13, 14, 15], 
            'cols': ['A - 832.CD', 'B - GESİ CAD. DOĞU', 'C - KADİR HAS BUL.', 'D - GESİ CAD BATI']
        },
        'İldem 1': {
            'locations': [16, 17, 18], 
            'cols': ['A - DİNÇER SOKAK', 'B - ALPARSLANTÜRKEŞ BUL.', 'D - GESİ CAD']
        },
        'İldem 2': {
            'locations': [19, 20, 21, 22], 
            'cols': ['A - DİNÇER SOKAK KUZEY', 'B - HANEDAN SOKAK', 'C - DİNÇER SOKAK GÜNEY', 'D - FETİH SOKAK']
        },
        'İldem 3': {
            'locations': [23, 24, 25], 
            'cols': ['A - DÜNDAR TAŞER CAD. KUZEY', 'C - DÜNDAR TAŞER CAD. GÜNEY', 'D - HANEDAN SOKAK']
        },
        'İldem 4': {
            'locations': [26, 27, 28, 29], 
            'cols': ['A - YAVUZSULTAN SELİM CAD. KUZEY', 'B - VATAN SOKAK', 'C - YAVUZSULTAN SELİM CAD. GÜNEY', 'D - ORKUN SOKAK']
        },
        'İldem 5': {
            'locations': [30, 31, 32, 33], 
            'cols': ['A - S.A BEDUK CAD. KUZEY', 'B - VATAN SOKAK BATI', 'C - S.A BEDUK CAD. GÜNEY', 'D - VATAN SOKAK DOĞU']
        }
    }
    
    # Başlangıç tarihi ve saati
    start_datetime = datetime.strptime("23.02.2025 05:00", "%d.%m.%Y %H:%M")
    
    # Her kavşak için ayrı sheet oluştur
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        
        for intersection_name, intersection_data in intersections.items():
            # Boş DataFrame oluştur
            result_df = pd.DataFrame()
            
            # Tarih ve saat sütunları
            tarih_list = []
            saat_list = []
            
            # Her timestep için satır oluştur
            unique_timesteps = sorted(df['timestep'].unique())
            
            for i, timestep in enumerate(unique_timesteps):
                current_time = start_datetime + timedelta(hours=i)
                tarih_list.append(current_time.strftime("%d.%m.%Y"))
                saat_list.append(current_time.strftime("%H:%M"))
            
            result_df['Tarih'] = tarih_list
            result_df['Saat'] = saat_list
            
            # Her kol için flow değerlerini ekle
            for col_idx, location in enumerate(intersection_data['locations']):
                col_name = intersection_data['cols'][col_idx]
                flow_values = []
                
                for timestep in unique_timesteps:
                    # İlgili timestep ve location için flow değerini bul
                    flow_value = df[(df['timestep'] == timestep) & (df['location'] == location)]['flow'].iloc[0]
                    flow_values.append(flow_value)
                
                result_df[col_name] = flow_values
            
            # Toplam sütunu ekle
            flow_cols = [col for col in result_df.columns if col not in ['Tarih', 'Saat']]
            result_df['Toplam'] = result_df[flow_cols].sum(axis=1)
            
            # Header ekle
            header_df = pd.DataFrame([['KBB RAPOR MODÜLÜ']], columns=[''])
            empty_df = pd.DataFrame([['']], columns=[''])
            
            # Final DataFrame oluştur
            final_df = pd.concat([header_df, empty_df, result_df], ignore_index=True)
            
            # Excel'e yaz
            final_df.to_excel(writer, sheet_name=intersection_name, index=False, header=False)
            
            # Worksheet'i al ve formatla
            worksheet = writer.sheets[intersection_name]
            
            # Sütun genişliklerini ayarla
            for column_cells in worksheet.columns:
                length = max(len(str(cell.value)) for cell in column_cells if cell.value)
                worksheet.column_dimensions[column_cells[0].column_letter].width = max(length + 2, 12)

def main():
    # Kullanım
    input_file = "test_gar.csv"
    output_file = "ildem_kış.xlsx"
    
    try:
        convert_to_intersection_format(input_file, output_file)
        print(f"Dönüştürme başarıyla tamamlandı! Çıktı dosyası: {output_file}")
        print("Oluşturulan sheet'ler: Gesi, Serkent, Beyazşehir, Toki, İldem 1-5")
        print(f"\nBaşlangıç tarihi: 23.02.2025 05:00 - {len(pd.read_csv(input_file)['timestep'].unique())} timestep işlendi")
    except FileNotFoundError:
        print(f"Hata: {input_file} dosyası bulunamadı!")
        print(f"Lütfen {input_file} dosyasının mevcut dizinde olduğundan emin olun.")
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()
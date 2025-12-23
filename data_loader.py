import pandas as pd
from datasets import Dataset
import os

class DataLoader:
    def __init__(self, file_path="IoT_Intrusion.csv"):
        self.file_path = file_path

    def load_data(self, sample_size=None):
        """
        Pandas kullanarak CSV'yi güvenli bir şekilde yükler ve 
        Hugging Face Dataset formatına çevirir.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"HATA: {self.file_path} dosyası bulunamadı!")

        print(f"[INFO] '{self.file_path}' dosyası Pandas ile okunuyor...")

        # Pandas okuma işlemi (Hata toleranslı)
        try:
            # Önce varsayılan utf-8 ile dene
            if sample_size:
                df = pd.read_csv(self.file_path, nrows=sample_size)
            else:
                df = pd.read_csv(self.file_path)
        except UnicodeDecodeError:
            print("[UYARI] UTF-8 hatası alındı, 'latin-1' deneniyor...")
            # Windows csv'leri genelde latin-1 veya cp1252 olur
            if sample_size:
                df = pd.read_csv(self.file_path, encoding='latin-1', nrows=sample_size)
            else:
                df = pd.read_csv(self.file_path, encoding='latin-1')
        except Exception as e:
            # Ayırıcı hatası olabilir (noktalı virgül mü?)
            print(f"[UYARI] Standart okuma başarısız: {e}")
            print("[INFO] Ayırıcı olarak ';' deneniyor...")
            if sample_size:
                df = pd.read_csv(self.file_path, sep=';', nrows=sample_size)
            else:
                df = pd.read_csv(self.file_path, sep=';')

        # Sütun isimlerini temizle (Başındaki/sonundaki boşlukları sil)
        # Örn: " Protocol Type " -> "Protocol Type"
        df.columns = df.columns.str.strip()

        print(f"[INFO] {len(df)} satır veri yüklendi.")
        print(f"[INFO] Tespit edilen sütunlar: {list(df.columns[:5])}...") # İlk 5 sütunu göster

        # Hugging Face Dataset formatına çevir
        dataset = Dataset.from_pandas(df)
        
        # Eğer Pandas index sütunu eklediyse onu kaldır (__index_level_0__)
        if "__index_level_0__" in dataset.column_names:
            dataset = dataset.remove_columns(["__index_level_0__"])

        return dataset
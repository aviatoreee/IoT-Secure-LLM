from transformers import AutoTokenizer, DataCollatorWithPadding
import torch

class IoTPreprocessor:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def row_to_sentence(self, batch):
        """
        IoT Veri setindeki özel sütunları siber güvenlik cümlelerine çevirir.
        """
        texts = []
        labels = []
        
        # Batch size hesapla (Sözlükteki herhangi bir listenin uzunluğu)
        batch_size = len(next(iter(batch.values())))
        
        for i in range(batch_size):
            features = []
            
            # --- 1. Temel Akış Bilgileri ---
            # 'flow_duration' ve 'Header_Length' gibi temel metrikler
            try:
                f_dur = batch.get('flow_duration', [0])[i]
                h_len = batch.get('Header_Length', [0])[i]
                rate = batch.get('Rate', [0])[i]
                tot_size = batch.get('Tot size', [0])[i]
                
                features.append(f"Flow duration is {f_dur}")
                features.append(f"Header length is {h_len}")
                features.append(f"Traffic rate is {rate}")
                features.append(f"Total size is {tot_size}")
            except Exception:
                pass # Sütun yoksa atla

            # --- 2. Protokol Tespiti (One-Hot Sütunlar) ---
            # Bu sütunlar genelde 1 veya 0 olur. Sadece 1 olanları cümleye ekliyoruz.
            active_protocols = []
            # Veri setindeki olası protokol sütunları
            proto_cols = ['HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']
            
            for proto in proto_cols:
                # Sütun varsa ve değeri 1 (veya 1.0) ise listeye ekle
                if proto in batch and float(batch[proto][i]) > 0:
                    active_protocols.append(proto)
            
            if active_protocols:
                features.append(f"Protocols used: {', '.join(active_protocols)}")

            # --- 3. Bayraklar (Flags) - Saldırı Tespiti İçin Kritik ---
            active_flags = []
            # Flag sütun isimleri 
            flag_map = {
                'fin_flag_number': 'FIN',
                'syn_flag_number': 'SYN',
                'rst_flag_number': 'RST',
                'psh_flag_number': 'PSH',
                'ack_flag_number': 'ACK',
                'ece_flag_number': 'ECE',
                'cwr_flag_number': 'CWR'
            }
            
            for col_name, flag_name in flag_map.items():
                if col_name in batch and float(batch[col_name][i]) > 0:
                    active_flags.append(flag_name)
            
            if active_flags:
                features.append(f"Flags set: {', '.join(active_flags)}")

            # --- 4. İstatistiksel Özellikler (Opsiyonel - Çok uzun olmasın diye seçici aldık) ---
            magnitude = batch.get('Magnitude', [0])[i] # Sütun adında yazım hatası var mı kontrol et: 'Magnitue' yazmışsın
            weight = batch.get('Weight', [0])[i]
            features.append(f"Magnitude is {magnitude}")
            features.append(f"Weight is {weight}")

            # --- Cümleyi Birleştir ---
            text = ". ".join(features) + "."
            texts.append(text)

            # --- 5. LABEL (Etiket) İşleme ---
            raw_label = batch['label'][i]
            
            # Etiketin formatına göre 0 veya 1'e çevirme
            # Eğer veri setinde 'Attack' yazıyorsa 1, başka bir şeyse 0 kabul eder.
            # CSV'de label sütunu 0 ve 1 ise zaten sorun yok.
            if isinstance(raw_label, str):
                # Örnek string kontrolü (Kendi verine göre düzenleyebilirsin)
                label_val = 1 if raw_label.lower() not in ['benigntraffic'] else 0
            else:
                # Zaten sayısal ise (0.0 veya 1.0 gibi)
                label_val = int(raw_label)
            
            labels.append(label_val)
            
        # Tokenize İşlemi
        tokenized_inputs = self.tokenizer(texts, truncation=True, max_length=128)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def prepare_data(self, dataset):
        print("[INFO] Veri işleniyor ve Tokenize ediliyor (Batched Process)...")
        
        # Orijinal sütunları kaldır ki RAM dolmasın
        original_columns = dataset.column_names
        
        tokenized_dataset = dataset.map(
            self.row_to_sentence,
            batched=True,
            batch_size=1000, 
            remove_columns=original_columns
        )
        
        print("[INFO] Train/Test ayrımı yapılıyor...")
        split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
        
        return split_dataset["train"], split_dataset["test"]
    
    def get_data_collator(self):
        return DataCollatorWithPadding(tokenizer=self.tokenizer)
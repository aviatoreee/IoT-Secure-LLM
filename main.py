import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

from src.data_loader import DataLoader
from src.preprocessor import IoTPreprocessor
from src.model_loader import get_model
from src.trainer import train_model

def plot_confusion_matrix(y_true, y_pred):
    """
    Modelin başarısını görselleştiren Karmaşıklık Matrisi (Confusion Matrix) çizer.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign (Normal)', 'Attack (Zararlı)'],
                yticklabels=['Benign (Normal)', 'Attack (Zararlı)'])
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.title('IoT Malware Tespit Sonuçları')
    plt.savefig('confusion_matrix.png')
    print("\n[GÖRSEL] 'confusion_matrix.png' dosyası kaydedildi.")

def simulate_attack(model, tokenizer, traffic_text):
    """
    Tekil bir trafik verisini modele sorar.
    """
    inputs = tokenizer(traffic_text, return_tensors="pt", truncation=True, max_length=128)
    
    # Eğer GPU varsa oraya taşı
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model = model.cuda()

    model.eval() # Modeli tahmin moduna al
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    confidence = torch.softmax(logits, dim=-1)[0][prediction].item()
    
    label_map = {0: "GÜVENLİ (BENIGN)", 1: "SALDIRI (ATTACK)"}
    print(f"\n--- Trafik Analizi ---")
    print(f"Girdi: {traffic_text[:100]}...") # İlk 100 karakter
    print(f"Sonuç: {label_map[prediction]}")
    print(f"Güven Oranı: %{confidence*100:.2f}")

def main():
    # 1. Veri Yükleme
    print("--- 1. Veri Yükleniyor ---")
    loader = DataLoader(file_path="datasets/IoT/IoT_Intrusion.csv")
    dataset = loader.load_data(sample_size=2000) # Test için 2000 satır

    # 2. Ön İşleme
    print("\n--- 2. Ön İşleme Yapılıyor ---")
    preprocessor = IoTPreprocessor()
    train_dataset, val_dataset = preprocessor.prepare_data(dataset)

    # 3. Model Hazırlığı
    print("\n--- 3. Model Hazırlanıyor ---")
    model = get_model()

    # 4. Eğitim
    print("\n--- 4. Eğitim Başlıyor ---")
    collator = preprocessor.get_data_collator()
    trainer = train_model(model, train_dataset, val_dataset, data_collator=collator)

    # 5. DETAYLI TEST VE ANALİZ
    print("\n--- 5. Sonuçlar Analiz Ediliyor ---")
    
    # Test seti üzerinde tahminler al
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    # Metrikleri yazdır
    print("\nDetaylı Metrikler:")
    print(predictions.metrics)

    # Görselleştirme
    plot_confusion_matrix(y_true, y_pred)

    # 6. Modeli Kaydetme
    print("\n--- 6. Model Kaydediliyor ---")
    trainer.save_model("./saved_iot_model")
    preprocessor.tokenizer.save_pretrained("./saved_iot_model")
    print("Model './saved_iot_model' klasörüne kaydedildi.")

    # 7. Manuel Simülasyon (Senaryo Testi)
    # Burada modelin dilini taklit eden sentetik bir cümle kuruyoruz.
    # Normalde bu cümle 'preprocessor.row_to_sentence' ile üretilir.
    fake_attack_log = "Flow duration is 0. Header length is 0. Protocols used: TCP. Flags set: SYN, FIN. Magnitude is 50. Weight is 100."
    simulate_attack(model, preprocessor.tokenizer, fake_attack_log)

if __name__ == "__main__":
    main()
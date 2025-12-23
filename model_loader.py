from transformers import AutoModelForSequenceClassification

def get_model(model_name="distilbert-base-uncased", num_labels=2):
    print(f"[INFO] {model_name} modeli {num_labels} sınıf için yükleniyor...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    return model
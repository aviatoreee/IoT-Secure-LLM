from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(model, train_dataset, val_dataset, data_collator):
    print("[INFO] Eğitim argümanları ayarlanıyor...")
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        
        # --- DÜZELTİLEN KISIM BURASI ---
        eval_strategy="steps",  # Eski adı: evaluation_strategy
        # -------------------------------
        
        save_strategy="no",
        fp16=False,
        use_cpu=False 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    print("[INFO] Eğitim başlıyor...")
    trainer.train()
    
    print("[INFO] Değerlendirme yapılıyor...")
    eval_result = trainer.evaluate()
    print(f"Sonuçlar: {eval_result}")
    
    return trainer
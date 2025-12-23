import tkinter as tk
from tkinter import scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import torch
import threading
import time
import transformers 
from transformers import AutoModelForSequenceClassification

# Senin yazdığın modülleri import ediyoruz
from src.preprocessor import IoTPreprocessor
from src.model_loader import get_model

class SecurityDashboard(ttk.Window):
    def __init__(self):
        super().__init__(themename="cyborg") # 'cyborg' teması siber güvenlik için havalı durur
        self.title("AI Tabanlı IoT Saldırı Tespit Sistemi")
        self.geometry("900x600")
        
        # Model ve Preprocessor'ı Yükle
        self.status_var = tk.StringVar(value="Sistem Başlatılıyor...")
        self.model = None
        self.preprocessor = None
        self.load_ai_components()

        # Arayüzü Oluştur
        self.create_widgets()

    def load_ai_components(self):
        """Modeli arka planda yükler ki arayüz donmasın"""
        def _load():
            self.preprocessor = IoTPreprocessor()
            # Eğittiğin kayıtlı modeli yüklemeye çalış, yoksa base modeli yükle
            try:
                #self.model = get_model() # Burada normalde kayıtlı model yolunu vermelisin
                self.model = AutoModelForSequenceClassification.from_pretrained("./saved_iot_model")
                self.model.eval()
                self.status_var.set("Sistem Hazır - İzleme Bekleniyor")
            except Exception as e:
                self.status_var.set(f"Model Yükleme Hatası: {str(e)}")
        
        threading.Thread(target=_load, daemon=True).start()

    def create_widgets(self):
        # --- ÜST PANEL (BAŞLIK) ---
        header_frame = ttk.Frame(self, padding=10)
        header_frame.pack(fill=X)
        ttk.Label(header_frame, text="🛡️ IoT NETWORK GUARDIAN", font=("Orbitron", 24, "bold"), bootstyle="info").pack()
        ttk.Label(header_frame, textvariable=self.status_var, font=("Consolas", 10), bootstyle="warning").pack()

        # --- ORTA BÖLÜM (2 SÜTUN) ---
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=BOTH, expand=True)

        # SOL PANEL: MANUEL GİRİŞ
        left_panel = ttk.Labelframe(main_frame, text="Manuel Trafik Analizi", padding=15)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=5)

        ttk.Label(left_panel, text="Trafik Logunu Yapıştır / Yaz:", font=("Arial", 10)).pack(anchor=W)
        self.input_text = tk.Text(left_panel, height=8, width=40, bg="#2b2b2b", fg="white", insertbackground="white")
        self.input_text.pack(fill=X, pady=5)
        self.input_text.insert("1.0", "Flow duration is 0. Protocols used: TCP. Flags set: SYN. Magnitude is 80.")

        analyze_btn = ttk.Button(left_panel, text="TEHDİT ANALİZİ BAŞLAT", command=self.analyze_traffic, bootstyle="danger-outline")
        analyze_btn.pack(fill=X, pady=10)

        # Sonuç Göstergesi
        self.result_label = ttk.Label(left_panel, text="SONUÇ: BEKLENİYOR", font=("Arial", 16, "bold"), bootstyle="secondary")
        self.result_label.pack(pady=20)
        
        self.confidence_bar = ttk.Progressbar(left_panel, value=0, length=200, bootstyle="success-striped")
        self.confidence_bar.pack(fill=X, pady=5)
        self.confidence_label = ttk.Label(left_panel, text="%0 Güven", font=("Arial", 9))
        self.confidence_label.pack()

        # SAĞ PANEL: GEÇMİŞ LOGLAR
        right_panel = ttk.Labelframe(main_frame, text="Tespit Geçmişi", padding=15)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=5)
        
        self.log_area = scrolledtext.ScrolledText(right_panel, height=20, width=40, state='disabled', bg="black", fg="#00ff00", font=("Consolas", 9))
        self.log_area.pack(fill=BOTH, expand=True)

    def analyze_traffic(self):
        """Girilen metni modele sorar"""
        if not self.model:
            return

        text_input = self.input_text.get("1.0", tk.END).strip()
        if not text_input:
            return

        # Tokenize ve Tahmin
        inputs = self.preprocessor.tokenizer(text_input, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
            probs = torch.softmax(logits, dim=-1)
            confidence = probs[0][prediction].item() * 100

        # Arayüzü Güncelle
        self.update_ui(prediction, confidence, text_input)

    def update_ui(self, prediction, confidence, text_input):
        result_text = "⚠️ SALDIRI TESPİT EDİLDİ!" if prediction == 1 else "✅ GÜVENLİ TRAFİK"
        style = "danger" if prediction == 1 else "success"
        
        # Sonuç Label
        self.result_label.config(text=result_text, bootstyle=style)
        
        # Progress Bar
        self.confidence_bar.config(value=confidence, bootstyle=style)
        self.confidence_label.config(text=f"%{confidence:.2f} Güven Skoru")

        # Loga Ekle
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {result_text} (Conf: %{confidence:.1f})\n> {text_input[:30]}...\n{'-'*40}\n"
        
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, log_entry)
        
        # Eğer saldırıysa logu kırmızı yap, değilse yeşil kalsın (Basit tag ekleme)
        if prediction == 1:
            # Son eklenen satırları kırmızı yapma işlemi eklenebilir
            pass
            
        self.log_area.see(tk.END) # En alta kaydır
        self.log_area.config(state='disabled')

if __name__ == "__main__":
    app = SecurityDashboard()
    app.mainloop()
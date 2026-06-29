import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import torch
import torch.nn as nn
from googletrans import Translator
from gtts import gTTS
import os
import platform
import pyttsx3
from langdetect import detect

# ── Tesseract path ─────────────────────────────────────────────────────────────
# FIX: set the path only on Windows; on Linux/macOS tesseract is on PATH already
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ── GRNN model ─────────────────────────────────────────────────────────────────
class GRNN(nn.Module):
    def __init__(self):
        super(GRNN, self).__init__()
        self.gru = nn.GRU(input_size=28, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 128)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# FIX: use weights_only=True (PyTorch >= 2.0) to silence the deprecation warning
grnn_model = GRNN()
grnn_model.load_state_dict(
    torch.load('grnn_model.pth', map_location=torch.device('cpu'), weights_only=True)
)
grnn_model.eval()
print(grnn_model)

# ── Text-to-speech engine ──────────────────────────────────────────────────────
engine = pyttsx3.init()

# ── Global state ───────────────────────────────────────────────────────────────
file_path = None   # FIX: declared at module level so all functions can access it safely
target_lang = None


# ── Core image processing ──────────────────────────────────────────────────────
def process_image(image_path):
    """Extract text with Tesseract OCR and append the GRNN digit prediction."""
    # OCR with Tesseract
    text = pytesseract.image_to_string(Image.open(image_path), lang='eng+fra')
    # FIX: 'fr' is not a valid Tesseract lang code — correct code is 'fra'

    # Prepare image for GRNN
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # FIX: use named constant
    img = cv2.resize(img, (28, 28))

    # Normalize with MNIST training stats
    img = (img.astype(np.float32) / 255.0 - 0.1307) / 0.3081
    img = np.reshape(img, (1, 28, 28))

    img_tensor = torch.from_numpy(img)
    with torch.no_grad():  # FIX: no gradient needed during inference
        output = grnn_model(img_tensor)

    digit = output.squeeze().argmax().item()
    return text, digit


# ── GUI callbacks ──────────────────────────────────────────────────────────────
def open_file_dialog(event=None):
    global file_path
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
    )
    if not path:   # FIX: user cancelled — do nothing instead of crashing
        return
    file_path = path

    img = Image.open(file_path)
    img = img.resize((400, 400))
    photo = ImageTk.PhotoImage(img)
    image_label.configure(image=photo)
    image_label.image = photo   # keep a reference so it isn't garbage-collected


def show_text():
    global target_lang
    if file_path is None:   # FIX: guard against no image being loaded
        messagebox.showerror("Erreur", "Veuillez d'abord importer une image.")
        return

    text, digit = process_image(file_path)
    lang = detect(text) if text.strip() else 'unknown'

    target_lang = None  # reset; user hasn't translated yet

    output_text.delete('1.0', tk.END)
    # FIX: was inserting text twice (once plain, once with lang tag)
    output_text.insert(tk.END, f"{text}\n[Digit GRNN: {digit}] [Langue détectée: {lang}]")


def on_closing():
    if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter ?"):
        root.destroy()


def translate_text(dest):
    global target_lang
    if file_path is None:
        messagebox.showerror("Erreur", "Veuillez d'abord importer une image.")
        return

    text, digit = process_image(file_path)

    try:
        translator = Translator()
        translated = translator.translate(text, dest=dest).text
    except Exception as e:
        messagebox.showerror("Erreur de traduction", str(e))
        return

    target_lang = dest
    output_text.delete('1.0', tk.END)
    output_text.insert(tk.END, f"{translated}\n[Digit GRNN: {digit}]")


def read_text():
    global target_lang
    if target_lang is None:
        messagebox.showerror("Erreur", "Veuillez d'abord afficher ou traduire le texte.")
        return
    if file_path is None:
        messagebox.showerror("Erreur", "Veuillez d'abord importer une image.")
        return

    text, _ = process_image(file_path)

    if target_lang != detect(text):
        try:
            translator = Translator()
            text = translator.translate(text, dest=target_lang).text
        except Exception as e:
            messagebox.showerror("Erreur de traduction", str(e))
            return

    try:
        tts = gTTS(text=text, lang=target_lang)
        tts.save("temp.mp3")
        # FIX: 'start' is Windows-only; use the right command per platform
        if platform.system() == 'Windows':
            os.system("start temp.mp3")
        elif platform.system() == 'Darwin':  # macOS
            os.system("open temp.mp3")
        else:  # Linux
            os.system("xdg-open temp.mp3")
    except Exception as e:
        messagebox.showerror("Erreur TTS", str(e))


# ── Build GUI ──────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("OCR avec Tesseract OCR et GRNN")
root.protocol("WM_DELETE_WINDOW", on_closing)

image_label = tk.Label(root)
image_label.pack(pady=5)

import_button = tk.Button(root, text="Importer une image", command=open_file_dialog)
import_button.pack(pady=2)

show_text_button = tk.Button(root, text="Afficher le texte", command=show_text)
show_text_button.pack(pady=2)

translate_to_arabic_button = tk.Button(
    root, text="Traduire en arabe", command=lambda: translate_text('ar')
)
translate_to_arabic_button.pack(pady=2)

translate_to_french_button = tk.Button(
    root, text="Traduire en français", command=lambda: translate_text('fr')
)
translate_to_french_button.pack(pady=2)

translate_to_english_button = tk.Button(
    root, text="Translate to English", command=lambda: translate_text('en')
)
translate_to_english_button.pack(pady=2)

read_text_button = tk.Button(root, text="Lire le texte", command=read_text)
read_text_button.pack(pady=2)

# Scrollable output text box
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

output_text = tk.Text(frame, font=("Helvetica", 14), yscrollcommand=scrollbar.set)
output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.config(command=output_text.yview)

# FIX: bind q only when root has focus, not globally, to avoid accidental quit
root.bind("<KeyPress-q>", lambda e: on_closing())

root.mainloop()

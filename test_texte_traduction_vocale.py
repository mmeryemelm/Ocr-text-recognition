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
import pyttsx3
from langdetect import detect

# Define the GRNN model
class GRNN(nn.Module):
    def __init__(self):
        super(GRNN, self).__init__()
        self.gru = nn.GRU(input_size=28, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 128, device='cpu')
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Load the GRNN model
grnn_model = GRNN()
grnn_model.load_state_dict(torch.load('grnn_model.pth', map_location=torch.device('cpu')))
grnn_model.eval()
print(grnn_model)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Define a function to process the image and extract the text using Tesseract OCR and GRNN
def process_image(image_path):
    # Extract the text from the image using Tesseract OCR
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(Image.open(image_path), lang='eng+fr')

    # Convert the image to grayscale and resize it for GRNN input
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (28, 28))

    # Normalize the image with the same values used during training
    img = (img.astype(np.float32) / 255.0 - 0.1307) / 0.3081

    # Reshape the image for GRNN input
    img = np.reshape(img, (1, 28, 28))

    # Convert the image to a tensor and pass it through the GRNN model
    img = torch.from_numpy(img)
    output = grnn_model(img)

    # Convert the output to text
    output = output.squeeze().argmax().item()

    return text + " " + str(output)

# Define a function to open a file dialog and import an image
def open_file_dialog(event=None):
    # Create a file dialog and allow the user to select a file
    global file_path
    file_path = filedialog.askopenfilename()

    # Display the selected image in the GUI
    img = Image.open(file_path)
    img = img.resize((400, 400))
    img = ImageTk.PhotoImage(img)
    image_label.configure(image=img)
    image_label.image = img

# Define a function to show the extracted text
def show_text():
    # Call process_image to extract the text from the image
    text = process_image(file_path)
    # Detect the language of the extracted text
    lang = detect(text)
    # Set the output text variable to display the extracted text
    global target_lang
    target_lang = None
    if 'ar' in output_text.get('1.0', tk.END):
        target_lang = 'ar'
    elif 'fr' in output_text.get('1.0', tk.END):
        target_lang = 'fr'
    elif 'en' in output_text.get('1.0', tk.END):
        target_lang = 'en'
    output_text.delete('1.0', tk.END)
    output_text.insert(tk.END, text)
    output_text.insert(tk.END, f"{text} ({lang})")

# Define a function to handle the "WM_DELETE_WINDOW" event
def on_closing():
    if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter ?"):
        root.destroy()

# Define a function to translate the extracted text to the specified language
def translate_text(dest):
    # Call process_image to extract the text from the image
    text = process_image(file_path)

    # Translate the text to the specified language
    translator = Translator()
    translated_text = translator.translate(text, dest=dest).text

    # Set the output text variable to display the translated text
    global target_lang
    target_lang = dest
    output_text.delete('1.0', tk.END)
    output_text.insert(tk.END, translated_text)

# Define a function to read the text aloud
def read_text():
    global target_lang
    if target_lang is None:
        messagebox.showerror("Erreur", "Veuillez d'abord afficher ou traduire le texte")
        return

    # Call process_image to extract the text from the image
    text = process_image(file_path)

    # Translate the text to the target language, if necessary
    if target_lang != 'en':
        translator = Translator()
        text = translator.translate(text, dest=target_lang).text

    # Read the text aloud using gTTS
    tts = gTTS(text=text, lang=target_lang)
    tts.save("temp.mp3")
    os.system("start temp.mp3")

# Create the GUI
root = tk.Tk()
root.title("OCR avec Tesseract OCR et GRNN")

# Bind the "WM_DELETE_WINDOW" event to the on_closing function
root.protocol("WM_DELETE_WINDOW", on_closing)

# Create a label to display the selected image
image_label = tk.Label(root)
image_label.pack()

# Create a button to open the file dialog and import an image
import_button = tk.Button(root, text="Importer une image", command=open_file_dialog)
import_button.pack()

# Create a button to show the extracted text
show_text_button = tk.Button(root, text="Afficher le texte", command=show_text)
show_text_button.pack()

# Create buttons to translate the extracted text to different languages
translate_to_arabic_button = tk.Button(root, text="Traduire en arabe", command=lambda: translate_text('ar'))
translate_to_arabic_button.pack()

translate_to_french_button = tk.Button(root, text="Traduire en français", command=lambda: translate_text('fr'))
translate_to_french_button.pack()

translate_to_english_button = tk.Button(root, text="Translate to English", command=lambda: translate_text('en'))
translate_to_english_button.pack()

# Create a button to read the text aloud
read_text_button = tk.Button(root, text="Lire le texte", command=read_text)
read_text_button.pack()

# Create a scrollbar and a text box to display the output text
scrollbar = tk.Scrollbar(root)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

output_text = tk.Text(root, font=("Helvetica", 20), yscrollcommand=scrollbar.set)
output_text.pack(side=tk.LEFT, fill=tk.BOTH)

scrollbar.config(command=output_text.yview)

# Bind the "q" key to the on_closing function
root.bind("<KeyPress-q>", on_closing)

# Start the GUI
root.mainloop()

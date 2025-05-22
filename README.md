# 🖼️ Image Caption Detection using CLIP

Unlock intelligent image understanding with **Image Caption Detection using CLIP** – a visually rich and AI-powered web app built using **Streamlit**, **OpenAI's CLIP model**, and **PyTorch**. Upload an image and discover the best-matching captions from a large set of creative descriptions.

## 🚀 Live Demo

Easily deploy on **Google Colab** and access via **ngrok**.
🔗 *https://b7e3-34-126-93-160.ngrok-free.app/*

---

## 📸 Features

* ⚡ **CLIP-powered caption similarity** using OpenAI's `clip-vit-base-patch32` model
* 🔁 Upload **multiple images** at once
* 🧠 Smart caption ranking based on **cosine similarity**
* 🧾 **Download matching captions** as CSV
* 🌍 Instant access using **ngrok tunnel**
* 💻 **Google Colab** and **Streamlit** compatible

---

## 🧰 Tech Stack

* Python 🐍
* [Streamlit](https://streamlit.io/) – interactive web UI
* [Transformers by Hugging Face](https://huggingface.co/docs/transformers/index)
* [PyTorch](https://pytorch.org/) – deep learning backend
* [CLIP Model](https://huggingface.co/openai/clip-vit-base-patch32)
* [ngrok](https://ngrok.com/) – to expose local Streamlit app to the web
* [Google Colab](https://colab.research.google.com/) – cloud development

---

## 🧪 How It Works

1. The uploaded image is processed using the CLIP model to extract **image features**.
2. Predefined **candidate captions** are embedded into **text features** using the same model.
3. Cosine similarity is computed between image and text features.
4. Captions are ranked from most to least relevant based on similarity.

---

## 🧾 Example Captions

* "A moment of bliss."
* "Nature’s silent poetry."
* "The taste of wanderlust."
* "Confidence looks good on me."

> More than **100+ creative and poetic captions** included for diverse scenarios like travel, nature, emotions, and lifestyle!

---

## 🛠️ How to Run in Google Colab

1. Upload `app.py` to your Colab session.
2. Set your `NGROK_AUTH_TOKEN` in environment:

   ```python
   import os  
   os.environ["NGROK_AUTH_TOKEN"] = "your_ngrok_token_here"
   ```
3. Run `app.py`:

   ```python
   !streamlit run app.py &
   ```
4. Follow the ngrok public URL to access your live app.

---

## 📂 Project Structure

```
📁 Image-Caption-Detection
│
├── app.py                # Main Streamlit application
├── clip_model/           # Saved CLIP model files
├── clip_processor/       # Saved CLIP processor files
└── README.md             # You're here!
```
---

## 💡 Future Improvements

* 🔤 Support dynamic caption generation using GPT-like models
* 🗂️ Caption clustering and tag suggestions
* 🌐 Multilingual caption support
* 📱 Mobile-friendly interface

---

## 🧑‍💻 Author

**Abhay Pratap Singh**
📧 [LinkedIn](https://www.linkedin.com/in/abhayjadon/) | 💼 AI & Data Science | 🎓 Final Year Engineer

---

## 📄 License

This project is licensed under the **MIT License** – feel free to use, modify, and share.


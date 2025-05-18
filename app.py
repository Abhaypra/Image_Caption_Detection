import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import io

# --- Load and save model/processor locally once ---
@st.cache_resource(show_spinner=False)
def load_model_and_save():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Save locally once (optional)
    model.save_pretrained("./clip_model")
    processor.save_pretrained("./clip_processor")
    return model, processor

model, processor = load_model_and_save()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Candidate captions ---
candidate_captions = [
    "Trees, Travel and Tea!",
    "A refreshing beverage.",
    "A moment of indulgence.",
    "The perfect thirst quencher.",
    "Your daily dose of delight.",
    "Taste the tradition.",
    "Savor the flavor.",
    "Refresh and rejuvenate.",
    "Unwind and enjoy.",
    "The taste of home.",
    "A treat for your senses.",
    "A taste of adventure.",
    "A moment of bliss.",
    "Your travel companion.",
    "Fuel for your journey.",
    "The essence of nature.",
    "The warmth of comfort.",
    "A sip of happiness.",
    "Pure indulgence.",
    "Quench your thirst, ignite your spirit.",
    "Awaken your senses, embrace the moment.",
    "The taste of faraway lands.",
    "A taste of home, wherever you are.",
    "Your daily dose of delight.",
    "Your moment of serenity.",
    "The perfect pick-me-up.",
    "The perfect way to unwind.",
    "Taste the difference.",
    "Experience the difference.",
    "A refreshing escape.",
    "A delightful escape.",
    "The taste of tradition, the spirit of adventure.",
    "The warmth of home, the joy of discovery.",
    "Your passport to flavor.",
    "Your ticket to tranquility.",
    "Sip, savor, and explore.",
    "Indulge, relax, and rejuvenate.",
    "The taste of wanderlust.",
    "The comfort of home.",
    "A journey for your taste buds.",
    "A haven for your senses.",
    "Your refreshing companion.",
    "Your delightful escape.",
    "Taste the world, one sip at a time.",
    "Embrace the moment, one cup at a time.",
    "The essence of exploration.",
    "The comfort of connection.",
    "Quench your thirst for adventure.",
    "Savor the moment of peace.",
    "The taste of discovery.",
    "The warmth of belonging.",
    "Your travel companion, your daily delight.",
    "Your moment of peace, your daily indulgence.",
    "The spirit of exploration, the comfort of home.",
    "The joy of discovery, the warmth of connection.",
    "Sip, savor, and set off on an adventure.",
    "Indulge, relax, and find your peace.",
    "A delightful beverage.",
    "A moment of relaxation.",
    "The perfect way to start your day.",
    "The perfect way to end your day.",
    "A treat for yourself.",
    "Something to savor.",
    "A moment of calm.",
    "A taste of something special.",
    "A refreshing pick-me-up.",
    "A comforting drink.",
    "A taste of adventure.",
    "A moment of peace.",
    "A small indulgence.",
    "A daily ritual.",
    "A way to connect with others.",
    "A way to connect with yourself.",
    "A taste of home.",
    "A taste of something new.",
    "A moment to enjoy.",
    "A moment to remember.",
    "Whispers of the wilderness.",
    "Nature’s silent poetry.",
    "Chasing sunlight and shadows.",
   "The earth laughs in flowers.",
   "Roots deep, spirit free.",
   "Breathe in the wild air.",
   "Where the sky meets the soul.",
     "Nature’s canvas, forever alive.",
     "Lost in the beauty of green.",
     "Waves, winds, and wild hearts.",
    "Just me, being me.",
    "Confidence looks good on me.",
    "Simplicity is the ultimate sophistication.",
    "Capturing my own sunshine.",
    "Embracing every imperfect moment.",
    "Radiate your own magic.",
    "My story, one smile at a time.",
    "Flaws and all, still fabulous.",
    "This is my vibe.",
    "Authenticity over everything."



]

# --- Function to do caption matching ---
def image_captioning(image_file, captions):
    image = Image.open(image_file).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_inputs = processor(text=captions, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()

    sorted_indices = similarities.argsort()[::-1]
    best_captions = [captions[i] for i in sorted_indices]
    sorted_similarities = similarities[sorted_indices]

    return best_captions, sorted_similarities

# --- Streamlit page config and CSS ---
st.set_page_config(page_title="Image Caption Matching", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .css-1d391kg {  /* Streamlit button */
        background-color: #4caf50;
        color: white;
    }
    .css-1d391kg:hover {
        background-color: #45a049;
        color: white;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🖼️ Image Caption Detection using CLIP")
st.write("Upload one or multiple images to see the best matching captions generated using a CLIP-based machine learning model.")

# Sidebar options
with st.sidebar:
    st.header("Options")
    top_n = st.slider("Number of top captions to show", 1, 10, 5)
    show_samples = st.checkbox("Show sample images", True)

if show_samples:
    st.markdown("### Sample Images")
    sample_images = [
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1494526585095-c41746248156?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?auto=format&fit=crop&w=800&q=80",
    ]
    cols = st.columns(len(sample_images))
    for idx, url in enumerate(sample_images):
        cols[idx].image(url, use_column_width=True)

uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for img_file in uploaded_files:
        st.markdown("---")
        st.image(img_file, caption=f"Uploaded Image: {img_file.name}", use_column_width=True)
        st.write("🔍 Matching best captions...")
        best_captions, similarities = image_captioning(img_file, candidate_captions)

        # Display results table
        results_df = pd.DataFrame({
            "Caption": best_captions[:top_n],
            "Similarity": similarities[:top_n]
        })
        st.table(results_df)

        # Download results CSV
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download captions as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{img_file.name}_captions.csv",
            mime="text/csv"
        )

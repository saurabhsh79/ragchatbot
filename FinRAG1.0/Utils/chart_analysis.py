import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

@st.cache_resource
def load_blip2_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_chart_caption(image: Image.Image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(output[0], skip_special_tokens=True)

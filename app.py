import streamlit as st
from streamlit_image_select import image_select
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline


# Project title 

st.title('Fake Job Posting Generator')

# Project banner image

st.image('banner_img_ml.jpeg')


def main():
    st.title("Fake Job Posting Predictor")
    st.markdown("fake_job_postings.csv")

    uploaded_file = st.file_uploader("fake_job_postings.csv", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(fake_job_postings.csv)
        st.write(df)

        # Load pre-trained model for text classification
        model = pipeline("text-classification", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased")

        predictions = []
        for index, row in df.iterrows():
            job_description = row["description"]
            prediction = model(job_description)[0]['label']
            predictions.append(prediction)

        df["prediction"] = predictions
        st.write("Predictions:", df["prediction"])

if __name__ == "__main__":
    main()

)


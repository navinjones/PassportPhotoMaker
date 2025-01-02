import streamlit as st
from rembg import remove
from PIL import Image
import os
from mtcnn import MTCNN
import numpy as np
import io

# Function to remove background
def remove_background(input_file):
    if isinstance(input_file, str):
        input_path = os.path.join('original', input_file)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The file {input_path} does not exist.")
        with open(input_path, 'rb') as f:
            input_img = f.read()
    else:
        input_img = input_file.read()

    output_path = 'masked/img_masked.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    subject = remove(input_img)
    with open(output_path, 'wb') as f:
        f.write(subject)

    return output_path

# Function to detect face and crop
def detect_face_and_crop(img_path):
    foreground_img = Image.open(img_path).convert("RGBA")
    detector = MTCNN()
    rgb_image = np.array(foreground_img.convert("RGB"))
    faces = detector.detect_faces(rgb_image)

    if faces:
        face = faces[0]
        x, y, w, h = face['box']
        h_pad = int(h * 0.1)
        w_pad = int(w * 0.1)
        lower_y = max(0, y - h_pad)
        upper_y = min(foreground_img.height, y + h + h_pad)
        lower_x = max(0, x - w_pad)
        upper_x = min(foreground_img.width, x + w + w_pad)

        face_crop = foreground_img.crop((lower_x, lower_y, upper_x, upper_y))
        return face_crop
    else:
        return None

# Function to resize and center image
def resize_and_center_image(img, target_size):
    img_aspect_ratio = img.width / img.height
    target_aspect_ratio = target_size[0] / target_size[1]

    if img_aspect_ratio > target_aspect_ratio:
        new_height = target_size[1]
        new_width = int(new_height * img_aspect_ratio)
    else:
        new_width = target_size[0]
        new_height = int(new_width / img_aspect_ratio)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    centered_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    centered_img.paste(resized_img, (paste_x, paste_y), resized_img)

    return centered_img

# Function to add a background color
def add_background(foreground, background_color, target_size):
    background = Image.new('RGB', target_size, color=background_color)
    background.paste(foreground, (0, 0), foreground)
    return background

# Main processing function
def process_image(input_file, background, output_path, target_size=(350, 450)):
    try:
        img_path = remove_background(input_file)
        cropped_img = detect_face_and_crop(img_path)

        if not cropped_img:
            return None

        resized_img = resize_and_center_image(cropped_img, target_size)

        if isinstance(background, str) and background.startswith('#'):
            final_img = add_background(resized_img, background, target_size)
        else:
            bg_path = os.path.join('bg', background)
            bg_img = Image.open(bg_path).resize(target_size, Image.LANCZOS)
            bg_img.paste(resized_img, (0, 0), resized_img)
            final_img = bg_img

        final_img.save(output_path, format='JPEG')
        return final_img

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit application
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Passport Photo Maker")

    mode = st.radio("Select Mode", ["Upload Photo", "Try Sample"], horizontal=True)
    target_size = (350, 450)

    if mode == "Upload Photo":
        uploaded_file = st.file_uploader("Upload your photo", type=['jpg', 'jpeg', 'png'])
        bg_color = st.color_picker("Select Background Color", "#FFFFFF")
    else:
        sample_images = [f for f in os.listdir('./original') if f.endswith(('jpg', 'png', 'jpeg'))]
        uploaded_file = st.selectbox("Select Sample Image", sample_images)
        bg_color = st.color_picker("Select Background Color", "#FFFFFF")

    if st.button("Submit"):
        if uploaded_file:
            output_path = "output.jpg"
            final_image = process_image(uploaded_file, bg_color, output_path, target_size)

            if final_image:
                st.image(final_image, caption="Processed Image")
                buf = io.BytesIO()
                final_image.save(buf, format="JPEG")
                st.download_button("Download Passport Photo", buf.getvalue(), "passport_photo.jpg", "image/jpeg")
            else:
                st.warning("No face detected in the uploaded image.")
        else:
            st.warning("Please upload or select an image.")

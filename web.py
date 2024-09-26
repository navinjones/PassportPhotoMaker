import streamlit as st
from rembg import remove
from PIL import Image
import os
from mtcnn import MTCNN
import numpy as np
import io

err_msg = None


def remove_background(input_file):
    if isinstance(input_file, str):
        input_path = os.path.join('original', input_file)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The file {input_path} does not exist.")
        with open(input_path, 'rb') as f:
            input_img = f.read()
    elif hasattr(input_file, 'name'):
        input_img = input_file.read()
    else:
        raise ValueError("Input must be a filename string or a file-like object with a 'name' attribute.")

    output_path = f'masked/img_maske.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    subject = remove(input_img, alpha_matting=True, alpha_matting_background_threshold=800)

    with open(output_path, 'wb') as f:
        f.write(subject)

    return output_path

def detect_face_and_crop(img_path):
    global err_msg
    if isinstance(img_path, str):
        foreground_img = Image.open(img_path).convert("RGBA")
        err_msg = "No face detected with face_recognition. Using MTCNN as fallback."

        detector = MTCNN()

        rgb_image = np.array(foreground_img.convert("RGB"))

        faces = detector.detect_faces(rgb_image)

        if faces:
            face = faces[0]
            x, y, w, h = face['box']
            h_pad = int(h * 0.9)
            w_pad = int(w * 0.9)
            lower_y = max(0, y - h_pad)
            upper_y = min(foreground_img.height, y + h + h_pad)
            lower_x = max(0, x - w_pad)
            upper_x = min(foreground_img.width, x + w + w_pad)

            face_crop = foreground_img.crop((lower_x, lower_y, upper_x, upper_y))
            return face_crop
        else:
            err_msg = "No face detected with either method. Returning original image."
            return None


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

    new_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_img.paste(resized_img, (paste_x, paste_y), resized_img)

    return new_img


def add_background(foreground, background_color, target_size):
    background = Image.new('RGB', target_size, color=background_color)
    background.paste(foreground, (0, 0), foreground)
    return background


def add_background1(foreground, background_file, target_size):
    background = Image.open(background_file).convert("RGBA")
    background = background.resize(target_size, Image.LANCZOS)
    background.paste(foreground, (0, 0), foreground)
    return background.convert("RGB")


def process_image(input_files, background, output_path, target_size=(350, 450)):
    processed_images = []

    if not isinstance(input_files, list):
        input_files = [input_files]

    for input_file in input_files:
        try:
            img_path = remove_background(input_file)

            if isinstance(input_file, str):
                img_name = input_file
            else:
                img_name = input_file.name

            cropped_img = detect_face_and_crop(img_path)
            if cropped_img is None:
                print(f"No face detected in {img_name}")
                return None
            resized_img = resize_and_center_image(cropped_img, target_size)

            if isinstance(background, str) and background.startswith('#'):
                final_img = add_background(resized_img, background, target_size)
            else:
                bg_path = os.path.join('bg', background)
                final_img = add_background1(resized_img, bg_path, target_size)

            output_file = f"{output_path}"
            final_img.save(output_file, format='jpeg')
            print(f'Image processing complete. Saved as {output_file}')

            processed_images.append(final_img)
        except FileNotFoundError as e:
            st.error(f"Error processing {input_file}: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred while processing {input_file}: {str(e)}")

    return processed_images


def clear_fields():
    st.session_state['text_input_key'] = str(st.session_state['text_input_key'])
    st.session_state['text_input_key'] += 'A'
    st.session_state['text_input_key1'] = str(st.session_state['text_input_key1'])
    st.session_state['text_input_key1'] += 'B'
    st.session_state['uploaded_files'] = []
    st.session_state['background'] = []
    st.session_state['color_wheel'] += 'D'
    st.experimental_rerun()


def load_images(image_directory):
    images = []
    for filename in os.listdir(image_directory):
        if filename.endswith(('png', 'jpg', 'jpeg')):
            images.append(filename)
    return images


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.markdown("<h1 style='text-align: center;'>Passport size filter</h1>", unsafe_allow_html=True)

    if 'text_input_key' not in st.session_state:
        st.session_state['text_input_key'] = 'A'

    if 'text_input_key1' not in st.session_state:
        st.session_state['text_input_key1'] = 'B'
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []
    if 'background' not in st.session_state:
        st.session_state['background'] = []
    if 'text_box' not in st.session_state:
        st.session_state['text_box'] = "C"
    if 'color_wheel' not in st.session_state:
        st.session_state['color_wheel'] = 'D'

    image_directory = './original'
    image_files = load_images(image_directory)
    back_image = "./bg"
    back_files = load_images(back_image)
    col1, col2 = st.columns([5, 5])

    with col1:
        st.header("Select Mode")
        search_option = st.radio("", ('new', 'existing'), horizontal=True, key=st.session_state['text_box'])
        output_image_path = 'output.jpg'

        if search_option == 'new':
            st.subheader("select input image")
            image = st.file_uploader(
                "Input image",
                type=['jpeg', 'jpg', 'png'],
                accept_multiple_files=False,
                help="Limit: 20MB per file",
                key=st.session_state['text_input_key']
            )
            st.subheader("select background color")
            bg_color = st.color_picker("Choose background color", "#ffffff", key=st.session_state['color_wheel'])

        elif search_option == 'existing':
            image = st.selectbox("**select input image:**", image_files)
            bg_color = st.selectbox('**select image:**', back_files)

        col3, col4 = st.columns([5, 1])
        is_submit = False
        with col3:
            if st.button('clear'):
                clear_fields()
        with col4:
            if st.button("submit"):
                is_submit = True
                if is_submit:
                    if image:
                        output_images = process_image(image, bg_color, output_image_path)
                        if output_images:
                            with col2:
                                output_container = st.container()
                                with output_container:
                                    for idx, img in enumerate(output_images):
                                        st.subheader(f"Processed Image")
                                        st.image(img)

                                        # Add download button
                                        buf = io.BytesIO()
                                        img.save(buf, format="JPEG")
                                        st.download_button(
                                            label="Download Passport Photo",
                                            data=buf.getvalue(),
                                            file_name=f"passport_photo_{idx + 1}.jpg",
                                            mime="image/jpeg"
                                        )
                        elif output_images is None:
                            with col2:
                                st.warning("No face detected in the uploaded image.")
                    else:
                        st.warning("Please upload or select at least one image.")

if __name__ == "__main__":
    main()

"""
Created on Sun Jul  4 20:36:16 2021

@author: Kudakwashe Binzi
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import os


@st.cache
def load_image(img):
    im = Image.open(img)
    return im


def save_uploaded_file(uploadedfile):
    if os.path.isfile(uploadedfile.name):
        os.remove(uploadedfile.name)
    else:
        print("Error: %s file not found" % uploadedfile.name)
    with open(os.path.join("", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())


def facial_detection(img_name):
    model = load_model('Model.h5')
    frame = cv2.imread(img_name)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in face:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        faces = face_cascade.detectMultiScale(roi_gray)
        if len(faces) == 0:
            print("Faces Not Detected")
        else:
            for (ex, ey, ew, eh) in faces:
                face_roi = roi_color[ey: ey + eh, ex:ex + ew]

    labeled = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite('/savedLabeled.png', labeled)
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

    final_image = cv2.resize(face_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0

    predictions = model.predict(final_image)

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

    return emotions[np.argmax(predictions)]


def main():
    """Facial Emotion Recognition App"""
    st.title("Facial Emotion Recognition App")
    st.text("Build with Streamlit and OpenCV")

    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Detection':
        st.subheader("Facial Emotion Recognition")

        upload = st.empty()
        with upload:
            image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            saved = save_uploaded_file(image_file)
            try:
                labeled_image = facial_detection(image_file.name)
                if st.button("Process"):
                    st.success("The Detected Emotion Is: " + labeled_image.upper())
                    if labeled_image == 'Happy':
                        st.subheader("YeeY! You look **_Happy_** :smile: today, always be! ")
                    elif labeled_image == 'Angry':
                        st.subheader("You seem to be **_Angry_** :rage: today, just take it easy! ")
                    elif labeled_image == 'Disgust':
                        st.subheader("You seem to be **_Disgusted_** :rage: today! ")
                    elif labeled_image == 'Fear':
                        st.subheader("You seem to be **_Fearful_** :fearful: today, be courageous! ")
                    elif labeled_image == 'Neutral':
                        st.subheader("You seem to be **_Neutral_** today, wish you a happy day! ")
                    elif labeled_image == 'Sad':
                        st.subheader("You seem to be **_Sad_** :worried: today, smile and be happy! ")
                    elif labeled_image == 'Surprised':
                        st.subheader("You seem to be **_Surprised_** today! ")

            except:
                if st.button("Process"):
                    st.error("No Emotions Detected! Make sure you upload an image which contains a face")

    elif choice == 'About':
        st.subheader(":smile: :worried: :fearful: :rage: :hushed:")
        st.markdown("____")
        st.subheader("**Done By**: Kudakwashe Binzi, Samuel Faindani and Denzel Makombe")
        st.markdown("**Dataset used for training:** https://www.kaggle.com/deadskull7/fer2013")
        st.markdown("**GitHub:** https://github.com/samie263/facial-emotion-recognition.git")
        st.markdown("**Video:** https://drive.google.com/drive/folders/1K65lcKE3zRQqpWMPafOKFAGL9kUnE1FF?usp=sharing")


if __name__ == '__main__':
    main()

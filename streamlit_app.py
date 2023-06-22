import streamlit as st

import time
import json

import mtcnn
import numpy as np
from PIL import Image
import io
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

# from streamlit.components.v1 import html
import requests
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_lottie import st_lottie
import joblib
from streamlit_option_menu import option_menu
import dlib
# Initialize state on first run
if (
    "btn_mesure" not in st.session_state
):  # check if the varaible exists as a session state
    st.session_state.btn_mesure = False  # store the variable outside of the app
if "btn_predir" not in st.session_state:
    st.session_state.btn_predir = False  # create a session


# load the model
@st.cache_resource  # this code i want to run it ones
def load_my_model():
    face_shape_model = tf.keras.models.load_model("vgg16-face-1")
    return face_shape_model

# Load the trained model (monture)
loaded_model_monture_gb = joblib.load('monture_gb_74_model.pkl')
# Load the label encoder dictionary
le_dict = joblib.load('le_dict.pkl')
loaded_model_materiaux=joblib.load('materiaux_model_99.pkl')

# dlib
def analyze_facial_features(image_array):
    # Initialize the face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "shape_predictor_81_facee_landmarks.dat"
    )
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = detector(gray)
    # Initialize the result dictionary
    result = {}
    # For each face detected
    for face in faces:
        # Detect landmarks
        landmarks = predictor(gray, face)
        # extract the point
        x_top = landmarks.part(71).x
        Y_top = landmarks.part(71).y
        cv2.circle(gray, (x_top, Y_top), 3, (0, 255, 225), -1)
        x_bottom = landmarks.part(8).x
        Y_bottom = landmarks.part(8).y
        cv2.circle(gray, (x_bottom, Y_bottom), 3, (0, 255, 225), -1)
        cv2.putText(
            gray,
            "D3",
            (int((x_top + x_bottom) / 2), Y_top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        height_face = math.sqrt((x_bottom - x_top) ** 2 + (Y_bottom - Y_top) ** 2)
        height_face = round(height_face, 2)
        # Draw a line D3
        cv2.line(gray, (x_top, Y_top), (x_bottom, Y_bottom), (0, 0, 255), 1)
        # extract forhead landmarks
        x_right_forehead = landmarks.part(75).x
        Y_right_forehead = landmarks.part(75).y
        cv2.circle(gray, (x_right_forehead, Y_right_forehead), 3, (0, 255, 225), -1)
        x_left_forehead = landmarks.part(79).x
        Y_left_forehead = landmarks.part(79).y
        cv2.circle(gray, (x_left_forehead, Y_left_forehead), 3, (0, 255, 225), -1)
        # Draw line D1 for forehead width
        cv2.line(
            gray,
            (x_left_forehead, Y_left_forehead),
            (x_right_forehead, Y_right_forehead),
            (0, 0, 255),
            1,
        )
        # Calculate forehead width
        forehead_width = math.sqrt(
            (x_right_forehead - x_left_forehead) ** 2
            + (Y_right_forehead - Y_left_forehead) ** 2
        )
        forehead_width = round(forehead_width, 2)
        cv2.putText(
            gray,
            "D2",
            (x_left_forehead - 50, int((Y_right_forehead + Y_left_forehead) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        # extract jaw landmarks
        x_right_jaw = landmarks.part(12).x
        y_right_jaw = landmarks.part(12).y
        cv2.circle(gray, (x_right_jaw, y_right_jaw), 3, (0, 255, 225), -1)
        x_left_jaw = landmarks.part(4).x
        y_left_jaw = landmarks.part(4).y
        cv2.circle(gray, (x_left_jaw, y_left_jaw), 3, (0, 255, 225), -1)
        cv2.line(
            gray, (x_left_jaw, y_left_jaw), (x_right_jaw, y_right_jaw), (0, 0, 255), 1
        )
        cv2.putText(
            gray,
            "D5",
            (int((x_left_jaw + x_right_jaw) / 2), int((y_left_jaw + y_right_jaw) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        # Calculate the width of the jaw
        jaw_width = math.sqrt(
            (landmarks.part(12).x - landmarks.part(4).x) ** 2
            + (landmarks.part(12).y - landmarks.part(4).y) ** 2
        )
        jaw_width = round(jaw_width, 2)
        # display d4  jawline length
        cv2.line(gray, (x_right_jaw, y_right_jaw), (x_bottom, Y_bottom), (0, 0, 255), 1)
        cv2.putText(
            gray,
            "D4",
            (int((x_right_jaw + x_bottom) / 2), int((y_right_jaw + Y_bottom) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        jawline_length = math.sqrt(
            (x_right_jaw - x_bottom) ** 2 + (y_right_jaw - Y_bottom) ** 2
        )
        jawline_length = round(jawline_length, 2)
        # extract distance between ears landmarks
        x_left_ear = landmarks.part(2).x
        y_left_ear = landmarks.part(2).y
        cv2.circle(gray, (x_left_ear, y_left_ear), 3, (0, 255, 225), -1)
        x_right_ear = landmarks.part(14).x
        y_right_ear = landmarks.part(14).y
        cv2.circle(gray, (x_right_ear, y_right_ear), 3, (0, 255, 225), -1)
        cv2.line(
            gray, (x_left_ear, y_left_ear), (x_right_ear, y_right_ear), (0, 0, 255), 1
        )
        cv2.putText(
            gray,
            "D1",
            (int((x_left_ear + x_right_ear) / 2), int((y_left_ear + y_right_ear) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        # calculate the disatnce between ears
        ear_distance = math.sqrt(
            (x_left_ear - x_right_ear) ** 2 + (y_left_ear - y_right_ear) ** 2
        )
        ear_distance = round(ear_distance, 2)
        # extract chin landmarks//menton
        x_chin_width_left = landmarks.part(6).x
        y_chin_width_left = landmarks.part(6).y
        x_chin_width = landmarks.part(10).x
        y_chin_width = landmarks.part(10).y
        cv2.circle(gray, (x_chin_width, y_chin_width), 3, (0, 255, 225), -1)
        cv2.circle(gray, (x_chin_width_left, y_chin_width_left), 3, (0, 255, 225), -1)
        cv2.line(
            gray,
            (x_chin_width, y_chin_width),
            (x_chin_width_left, y_chin_width_left),
            (0, 0, 255),
            1,
        )
        cv2.putText(
            gray,
            "D6",
            (
                int((x_chin_width + x_chin_width_left) / 2),
                int((y_chin_width + y_chin_width_left) / 2),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        chin_width = math.sqrt(
            (x_chin_width - x_chin_width_left) ** 2
            + (y_chin_width - y_chin_width_left) ** 2
        )
        chin_width = round(chin_width, 2)
        # extract jaw landmarks//machoire
        x_jaw = landmarks.part(7).x
        y_jaw = landmarks.part(7).y
        cv2.circle(gray, (x_jaw, y_jaw), 3, (0, 255, 225), -1)
        x_jaw_left = landmarks.part(9).x
        y_jaw_left = landmarks.part(9).y
        cv2.circle(gray, (x_jaw_left, y_jaw_left), 3, (0, 255, 225), -1)
        cv2.line(gray, (x_jaw, y_jaw), (x_jaw_left, y_jaw_left), (0, 0, 255), 1)
        shape_chin = math.sqrt((x_jaw - x_jaw_left) ** 2 + (y_jaw - y_jaw_left) ** 2)
        shape_chin = round(shape_chin, 2)
        # Define the vectors corresponding to D3 and D4
        D3 = np.array([x_bottom - x_top, Y_bottom - Y_top])
        D4 = np.array([x_bottom - x_right_jaw, Y_bottom - y_right_jaw])
        # Calculate the dot product of D3 and D4
        dot_product = np.dot(D3, D4)
        # Calculate the magnitudes of D3 and D4//grandeur
        magnitude_D3 = np.linalg.norm(D3)
        magnitude_D4 = np.linalg.norm(D4)
        # Calculate the cosine of the angle between D3 and D4
        cos_theta = dot_product / (magnitude_D3 * magnitude_D4)
        # Calculate the angle in degrees
        angle_degreesA2 = np.degrees(np.arccos(cos_theta))
        # Define the vectors corresponding to the landmarks
        D1 = np.array([x_left_ear - x_right_ear, y_left_ear - y_right_ear])
        D5 = np.array([x_right_jaw - x_right_ear, y_right_jaw - y_right_ear])
        # Calculate the dot product of D1 and D5
        dot_product = np.dot(D1, D5)
        # Calculate the magnitudes of D1 and D5
        magnitude_D1 = np.linalg.norm(D1)
        magnitude_D5 = np.linalg.norm(D5)
        # Calculate the cosine of the angle between D1 and D5
        cos_theta = dot_product / (magnitude_D1 * magnitude_D5)
        # Calculate the angle in degrees
        angle_degreesA3 = np.degrees(np.arccos(cos_theta))
        # Define the vectors corresponding to the lines
        D3 = np.array([x_top - x_bottom, Y_top - Y_bottom])
        D6 = np.array([x_chin_width - x_bottom, y_chin_width - Y_bottom])
        # Calculate the dot product of D3 and D6
        dot_product = np.dot(D3, D6)
        # Calculate the magnitudes of D3 and D6
        magnitude_D3 = np.linalg.norm(D3)
        magnitude_D6 = np.linalg.norm(D6)
        # Calculate the cosine of the angle between D3 and D6
        cos_theta = dot_product / (magnitude_D3 * magnitude_D6)
        # Calculate the angle in degrees
        angle_degreesA1 = np.degrees(np.arccos(cos_theta))
        # Add forehead width to the result dictionary
        result["height_face D3"] = height_face
        result["forehead_width D2"] = forehead_width
        result["ear_distance D1"] = ear_distance
        result["jaw_width D5"] = jaw_width
        result["chin_width D6"] = chin_width
        result["jawline_length D4"] = jawline_length
        result["shape_chin D7"] = shape_chin
        result["Angle A1"] = angle_degreesA1
        result["Angle A2"] = angle_degreesA2
        result["Angle A3"] = angle_degreesA3
        st.image(gray, channels="RGB", use_column_width=True)
    return result


def crop_and_resize(image, target_w=224, target_h=224):
    """this function crop & resize images to target size by keeping aspect ratio"""
    if image.ndim == 2:
        img_h, img_w = image.shape  # for Grayscale will be   img_h, img_w = img.shape
    elif image.ndim == 3:
        (
            img_h,
            img_w,
            channels,
        ) = image.shape  # for RGB will be   img_h, img_w, channels = img.shape
    target_aspect_ratio = target_w / target_h
    input_aspect_ratio = img_w / img_h
    if input_aspect_ratio > target_aspect_ratio:
        resize_w = int(input_aspect_ratio * target_h)
        resize_h = target_h
        img = cv2.resize(image, (resize_w, resize_h))
        crop_left = int((resize_w - target_w) / 2)  ## crop left/right equally
        crop_right = crop_left + target_w
        new_img = img[:, crop_left:crop_right]
    if input_aspect_ratio < target_aspect_ratio:
        resize_w = target_w
        resize_h = int(target_w / input_aspect_ratio)
        img = cv2.resize(image, (resize_w, resize_h))
        crop_top = int(
            (resize_h - target_h) / 4
        )  ## crop the top by 1/4 and bottom by 3/4 -- can be changed
        crop_bottom = crop_top + target_h
        new_img = img[crop_top:crop_bottom, :]
    if input_aspect_ratio == target_aspect_ratio:
        new_img = cv2.resize(image, (target_w, target_h))
    return new_img


detector = MTCNN()


def extract_face(img, target_size=(224, 224)):
    # 1. detect faces in an image
    results = detector.detect_faces(img)
    if (
        results == []
    ):  # if face is not detected, call function to crop & resize by keeping aspect ratio
        new_face = crop_and_resize(img, target_w=224, target_h=224)
    else:
        x1, y1, width, height = results[0]["box"]
        x2, y2 = x1 + width, y1 + height
        face = img[
            y1:y2, x1:x2
        ]  # this is the face image from the bounding box before expanding bbox
        # 2. expand the top & bottom of bounding box by 10 pixels to ensure it captures the whole face
        adj_h = 10
        # assign value of new y1
        if y1 - adj_h < 10:
            new_y1 = 0
        else:
            new_y1 = y1 - adj_h
        # assign value of new y2
        if y1 + height + adj_h < img.shape[0]:
            new_y2 = y1 + height + adj_h
        else:
            new_y2 = img.shape[0]
        new_height = new_y2 - new_y1
        # 3. crop the image to a square image by setting the width = new_height and expand the box to new width
        adj_w = int((new_height - width) / 2)
        # assign value of new x1
        if x1 - adj_w < 0:
            new_x1 = 0
        else:
            new_x1 = x1 - adj_w
        # assign value of new x2
        if x2 + adj_w > img.shape[1]:
            new_x2 = img.shape[1]
        else:
            new_x2 = x2 + adj_w
        new_face = img[
            new_y1:new_y2, new_x1:new_x2
        ]  # face-cropped square image based on original resolution
    # 4. resize image to the target pixel size
    sqr_img = cv2.resize(new_face, target_size)
    return sqr_img


y_label_dict = {0: "cœur", 1: "oblong", 2: "ovale", 3: "rond", 4: "carré"}


def predict_face_shape(image_array):
    # first extract the face using bounding box
    face_img = extract_face(
        image_array
    )  # call function to extract face with bounding box
    new_img = cv2.cvtColor(
        face_img, cv2.COLOR_BGR2RGB
    )  # convert to RGB -- use this for display
    # convert the image for modelling
    test_img = np.array(new_img, dtype=float)
    test_img = test_img / 255
    test_img = np.array(test_img).reshape(1, 224, 224, 3)
    # make predictions
    face_shape_model = load_my_model()
    pred = face_shape_model.predict(test_img)
    label = np.argmax(pred, axis=1)
    shape = y_label_dict[label[0]]
    pred = np.max(pred)
    pred = np.around(pred * 100, 2)
    return shape, pred, new_img


def make_prediction_monture(genre, select_type,shape, select_style, select_uti):
    input_data = {
        'genre': [genre],
        'type.type': [select_type],
        'visage.visage': [shape],
        'style': [select_style],
        'utilisation': [select_uti]
    }
    input_df = pd.DataFrame(input_data)
    st.write("la forme de votre  visage est ",shape)
    # Encode input features
    for column in input_df.columns:
        le = le_dict[column]
        input_df[column] = le.transform(input_df[column])

    # Make prediction
    prediction_monture = loaded_model_monture_gb.predict(input_df)
    prediction_materaiux=loaded_model_materiaux.predict(input_df)
    #st.write("la monture faite pour vous est  ", prediction[0])
    return prediction_monture[0],prediction_materaiux[0]










def main():
    # page configuration
    st.set_page_config(
        page_title="KatYos Virtual Assistant",
        page_icon="💬,🤖",
    )

    # to remove stremlit app in the footer and humburger
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=[
                "Home",
                "Tips",
                "face shape detection",
                "upload photo",
                "using video",
                "complete profile",
            ],
            icons=[
                "house",
                "card-checklist",
                "skip-start-circle-fill",
                "cloud-arrow-up",
                "camera-fill",
                "chat-right-text",
            ],
            default_index=0,  # when we run the app which option should be selected first,
            styles={
                "container": {"background-color": "#f2fefe "},
                "icon": {"color": "#3e07b4"},
                "nav-link-selected": {"background-color": "#06f3ff"},
            },
        )

    def home():
        
        st.markdown(
            "<h1 style='text-align:center;'> visagisme Virtuel  </h1>",
            unsafe_allow_html=True,
        )
        st.markdown("------")
        # to design the rectangle shadow
        st.write(
            """<div style='
    background-color: #3e07b4;
    padding: 20px;border-radius: 
    5px;box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
'><h2 style='text-align: center; color: white; font-size: 35px;'>On vous aide à trouver la monture faite pour vous!</h2>
</div>
""",
            unsafe_allow_html=True,
        )

        with open("help.json") as source:
            animation = json.load(source)  # for the annimation help

        col1, col2, col3 = st.columns(3)
        with col3:
            st_lottie(animation, width=300, height=300)
        with col1:
                    
            btn = st.markdown("""
                        <style>
                        div.stButton > button:first-child {
                            background-color: #8fdeed;
                            color:#3e07b4;
                            font-size:25px;
                            height:3em;
                            margin-top:90px;
                            
                        position:relative;rigth:50%;
                            width:15em;
                            border-radius:0.75rem;
                           
                        }
                        div.stButton > button:hover {
                        background-color: #06f3ff;
                        color:#ffffff;
                        border: 2px solid white;
                        
   
                        
                }       </style>""", unsafe_allow_html=True)
            btn = st.button("Before we Start")
            #if btn:
                
                #st.sidebar.markdown("You clicked the button!")                
                  
                
           
    def tips():
        
        st.title("     Quelques conseils avant de commencer     ")
        st.markdown("------")
        st.header("1 Placer toi dans un endroit bien éclairé")
        st.header("2 Regarder droit vers la caméra")
        st.header("3 Garder le visage fixe et ne souriez pas")
        st.header("4 Enlever vos lunettes si vous en portez")
        st.header("5 éloingner vos cheveux de votre visage")
        btn = st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #8fdeed;
                    color:#3e07b4;
                        font-size:25px;
                        height:3em;
                        width:15em;
                        margin-top:40px;
                        border-radius:0.75rem;
                    }
                    div.stButton > button:hover {
                    background-color: #06f3ff;
                    color:#ffffff;
                    border: 2px solid white;                    
            }       </style>""", unsafe_allow_html=True)
        btn_start=st.button("Get Start")

    def face_shape_detection():
        st.header("upload a front-facing photo or take a photo so our IA can determine your face shape")
        st.markdown("------")
        col1, col2, col3 =st.columns(3)

        
        with open("face-recognition.json") as source:
            face = json.load(source)  # for the annimation help

    
            st_lottie(face, width=500, height=500)
        col1,col2,col3=st.columns(3)
        btn = st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #8fdeed;
                    color:#3e07b4;
                        font-size:25px;
                        height:3em;
                        width:15em;
                        border-radius:0.75rem;
                        
                    }
                    div.stButton > button:hover {
                    background-color: #06f3ff;
                    color:#ffffff;
                    border: 2px solid white;                    
            }       </style>""", unsafe_allow_html=True)
        button_photo=col1.button("upload a photo")
        button_video=col3.button("web cam")
        if button_photo:
            upload_photo()
        elif button_video:
            using_video()

    if 'image_array' not in st.session_state:
        st.session_state['image_array'] = None 
    def upload_photo():
        
        image = st.file_uploader(
            "Veuillez télécharger une photo de profil",
            type=["png", "jpg", "jpeg", "bmp", "svg"],
        )
        if image is not None:
            st.image(image)
            pil_image = Image.open(io.BytesIO(image.read()))
            image_array = np.array(pil_image)

             # Initialize and store image_array in session state
             # Clear cv2_img from session state if it exists
            
            st.session_state.image_array = image_array

            btn_mesure = st.button("donner les mesures", key="id_photo")
            if btn_mesure:
                st.session_state.btn_mesure = True
                result = analyze_facial_features(image_array)
                # Convert the result dictionary to a dataframe
                df = pd.DataFrame.from_dict(result, orient="index", columns=["Value"])
                st.write("Analysis Result")
                # if st.checkbox("Analysis Result"):
                st.table(df)
            btn_predir = st.button("predir face shape", key="predir_photo")
            if btn_predir:
                st.session_state.btn_predir = True
                shape, pred, new_img = predict_face_shape(image_array)
                st.image(new_img, caption=" bounding box on extracted face")
                st.write(
                    "la forme de votre visage est :",
                    shape,
                    "  avec une probabilité de :",
                    pred,
                )
                
    import av
    def rectangle(frame:av.VideoFrame) -> av.VideoFormat:
        frame_data = frame.to_ndarray(format='rgb24')
        height, width, _ = frame_data.shape

        rectangle_width = 200  # Width of the rectangle
        rectangle_height = 200  # Height of the rectangle
        rectangle_top_left = ((width - rectangle_width) // 2, (height - rectangle_height) // 2)
        rectangle_bottom_right = ((width + rectangle_width) // 2, (height + rectangle_height) // 2)
    # Create a mask to cover the entire frame
        mask = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw a filled rectangle on the mask
        cv2.rectangle(mask, rectangle_top_left, rectangle_bottom_right, (0, 0, 0), -1)

        # Apply the mask to the frame to make the area outside the rectangle opaque
        blended_frame = cv2.addWeighted(frame_data, 1.0, mask, 0.3, 0)



        # Draw the non-opaque rectangle on the frame
        rectangle_color = (221, 221, 221)  # Gray color
        rectangle_thickness = 1
        cv2.rectangle(blended_frame, rectangle_top_left, rectangle_bottom_right, rectangle_color, rectangle_thickness)

        # Convert the numpy array back to a video frame
        modified_frame = av.VideoFrame.from_ndarray(blended_frame, format='rgb24')
    
        return modified_frame
    def using_video():
        from streamlit_webrtc import webrtc_streamer        

        #webrtc_streamer(key="sample", media_stream_constraints={"video": True, "audio": False},video_frame_callback=rectangle) 
    
         


        camera_photo = st.camera_input("Trouvez la forme de votre visage", label_visibility="hidden")
        
        if camera_photo is not None:
            bytes_data = camera_photo.getvalue()
            
            cv2_img = cv2.imdecode(
                np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
            )
           
            st.session_state.cv2_img=cv2_img
              
                
           
            
            
            result = analyze_facial_features(cv2_img)

            
        btn_mesure = st.button("donner les mesures", key="id_video")
        if btn_mesure:
            st.session_state.btn_mesure = True
            # Convert the result dictionary to a dataframe
            
            df = pd.DataFrame.from_dict(result, orient="index", columns=["Value"])
            st.table(df)
        btn_predir = st.button("predir face shape", key="predir_video")
        if btn_predir:
            st.session_state.btn_predir = True
            shape, pred, new_img = predict_face_shape(cv2_img)
            st.image(new_img, caption=" bounding box on extracted face")
            st.write(
                "la forme de votre visage est :",
                shape,
                "  avec une probabilité de :",
                pred,
            )

    def complete_profile():
    
        #radio_btn = st.radio("êtes-vous..", options=("femme", "homme"))
        genre = st.selectbox('Genre', ['homme', 'femme'])
        select_type = st.selectbox(
            "Quel type de lunettes recherchez-vous ?", options=("de soleil", "de vue")
        )

        select_style = st.selectbox(
            "Quel style de lunettes recherchez-vous ?",
            options=("unique",
                "élégant",
                "sophistiqué",
                "sportif",
                "unique",
                "jeune",
                "décontractée",
                "sérieux",
                "dynamique",
                "chic",
                "classique",
                " branché ",
                "rétro",
                "audacieux",
                "intellectuel",
                "vinatge",
                "rock",
                "tendance",
                "mode",
                " branché ",
                "sympathique",
                "intemporel",
                "audacieux",
                "féminine",
                "estival",
                "coquine",
                "coquette",
            ),
        )
        select_uti = st.selectbox(
            "Pour quel usage ?",
            options=(
                "activites sportif",
                "plage",
                
                "activité nautique",
                "usage quotidien",
                "activités professionnelles",
                "conduite",
                "sortie en ville",
                "activité en plein air",
                "occasion spécial",
            ),
              # Perform prediction

        )
        
        
        # Get the image_array from session_state
        if 'cv2_img' in st.session_state:
            cv2_img = st.session_state.cv2_img
            shape, _, _ = predict_face_shape(cv2_img)
            
            
           

        elif  'image_array' in st.session_state:
            
            image_array = st.session_state.image_array#appler variable
            shape, _, _ = predict_face_shape(image_array)
            
            
            
           

        pre=st.button('Save prediction')
        
        if pre : 
            st.session_state.clear()
            #if 'cv2_img' in st.session_state:
            #    del st.session_state['cv2_img']
            #else:
             #   del st.session_state['image_array']
            prediction_monture,prediction_materaiux=make_prediction_monture(genre, select_type, shape, select_style, select_uti)
            st.write("la monture faite pour vous est : ", prediction_monture)

            encoded_select_type = ""
            valid_formes_soleil_katyos = ["Aviator", "Carrées", "Masque", "Oeil de chat", "Ovales", "Papillon", "Pilote", "Rectangulaires", "Rondes"]
            valid_formes_de_vue_katyos = ["Carrées", "Oeil de chat", "Ovales", "Papillon", "Pilote", "Rectangulaires", "Rondes"]
            encoded_select_type = select_type.replace(" ", "-")

            if select_type == "de soleil" and prediction_monture in valid_formes_soleil_katyos:
                
                url = f"https://katyos.com/6-lunettes-{encoded_select_type}-?q=Formes-{prediction_monture}"
                #st.subheader("vous pouver trouver votre monture dans le notre mall :",url)
           
                st.subheader("Vous pouvez trouver votre monture dans notre mall :")
                message = f"[Cliquez ici pour accéder au site]({url})"
                st.markdown(message, unsafe_allow_html=True)

           
            elif select_type == "de vue" and prediction_monture in valid_formes_de_vue_katyos:
                
                
                url = f"https://katyos.com/3-lunettes-{encoded_select_type}-?q=Formes-{prediction_monture}"
                #st.subheader("vous pouver trouver votre monture dans le notre mall : " ,url)
                                
                st.subheader("Vous pouvez trouver votre monture dans notre mall :")
                message = f"[Cliquez ici pour accéder au site]({url})"
                st.markdown(message, unsafe_allow_html=True)
            else:
                
        
                error_message = "Cette monture n'est pas encore disponible dans notre site ,Veuillez visiter notre site ultérieurement."
                st.subheader(error_message)


            #else:
            #    url = f"https://katyos.com/3-lunettes-{encoded_select_type}-?q=Formes-{prediction}"

            #st.subheader(url)
            st.write(f"Pour votre usage : { select_uti } ,il est recommandé d'utiliser des montures qui sont fabriqués avec l'un de ces materiaux : " , prediction_materaiux)

           

        btn = st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #8fdeed;
                    color:#3e07b4;
                        font-size:25px;
                        height:3em;
                        width:15em;
                        margin-top:40px;
                        border-radius:0.75rem;
                    }
                    div.stButton > button:hover {
                    background-color: #06f3ff;
                    color:#ffffff;
                    border: 2px solid white;                    
            }       </style>""", unsafe_allow_html=True)
       # btn_predict = st.button("visiter notre mall virtuel")
        
        #add session state
        #if btn_predict:
          
    page_names_to_funcs = {
        "Home": home,
        "Tips": tips,
        "face shape detection": face_shape_detection,
        "upload photo": upload_photo,
        "using video": using_video,
        "complete profile":  complete_profile,
    }
    page_names_to_funcs[selected]()  # appeler la fonction corres


#    if selected=="Home":


if __name__ == "__main__":
    main()

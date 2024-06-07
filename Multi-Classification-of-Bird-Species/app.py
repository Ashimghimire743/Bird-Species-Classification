import streamlit as st
import numpy as np
import shutil
import wikipediaapi
import os
import gdown

from streamlit_option_menu import option_menu
from streamlit_cropper import st_cropper
from PIL import Image
from predict import img_preprocess,pred_label

st.set_page_config(
    page_title="AvianVision",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="static/Bird1.png",
)
if os.path.exists('static/tutorial_Video.mp4'):
    pass
else:
    gdown.download('https://drive.google.com/uc?id=1svlishBOwiBin5smpHbkjFpgvCJNqETe',output='static/tutorial_Video.mp4',quiet=False)


#============================================================================================================================
#Option Menu
with st.sidebar:
    col1,col2 = st.columns([0.3,0.8])
    with col1:
        pass
    with col2:
        st.image("static/Bird1.png",width=100)
    selected = option_menu('AvianVision', ["Intro", 'Upload','Result','About'], 
        icons=['play-btn','upload','check-circle','info-circle'],menu_icon='intersect', default_index=0)
    
#============================================================================================================================
#Intro
if selected=="Intro":
    #Header
    st.title('Welcome to AvianVision')
    st.subheader('*Your go-to destination for seamless bird identification and exploration*')

    st.divider()

    with st.container():
        col1,col2=st.columns([0.65, 0.35])
        with col1:
            st.header('Use Cases')
            st.markdown(
                """
                -  _Have no idea on the bird you saw?_

                - _Looking to conduct research on bird species?_

                - _Looking for a tools?_

                - _Just here to play and learn?_

                """
                )
        with col2:
            st.image('static/Parrot.png',width=250)

    st.divider()
    st.header("Tutorial Video")
    st.video("static/tutorial_Video.mp4")

#============================================================================================================================
#Upload
saved_images = len(os.listdir('static/save')) if os.path.exists('static/save') else 0
class_name,confidence = None,None
def save_images_list():
    if saved_images:
        col = st.columns(10)
        for i in range(0, saved_images):
            with col[i]:
                image = Image.open(f"static/save/cropped_image{i}.png")
                width = image.thumbnail((600,600))
                st.image(f"static/save/cropped_image{i}.png", width=width)
                st.write(i+1)
    else:
        st.write('saved images not found')

if selected=="Upload":
    #Header
    st.title('Upload')
    st.subheader('*Your go-to destination for seamless bird identification and exploration*')

    st.divider()    
    #uploading files

    uploaded_file = st.file_uploader("Choose a file", type=["jpeg","jpg","png"])
    if uploaded_file:
        st.header('Crop your picture')
        st.write('Is there multiple birds in you picture, just Drag and Save')
            
        col1, col2 = st.columns(2)
        with col1:
            col1.subheader('Original')
            image = Image.open(uploaded_file)
            image.thumbnail((600, 600))
            cropped_image = st_cropper(image, realtime_update=True)
            crop_btn = st.button("Save")

        with col2:
            col2.subheader("Preview")
            if crop_btn and cropped_image:
                st.image(cropped_image, use_column_width=True)  # Display cropped image for preview
                if not os.path.exists('static/save'):
                    os.makedirs('static/save')
                if saved_images<10:
                    cropped_image.save(f"static/save/cropped_image{saved_images}.png")
                    st.write(f"No of images saved: {saved_images+1}, :red[[limit: 10]]")
                else:
                    st.write(":red[limit reached, [limit: 10]]")

        st.divider()
        st.subheader('Show your Saved Images')
        done = st.button("Yes")
        clear = st.button('Clear')
        if done:
            save_images_list()
        if clear:
            shutil.rmtree('static/save')
            st.write('You can save new images now')
        st.divider()
        st.write('Note: after saving your images go to Results')


#============================================================================================================================
#Result
def get_bird_info(bird_name):
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='BirdSpeciesClassification (tanayastha2000@gmail.com)',extract_format=wikipediaapi.ExtractFormat.WIKI)  # Initialize Wikipedia API

    # Retrieve page object for the specified bird name
    page = wiki_wiki.page(bird_name, unquote=True)

    if page.exists():
        # Extract summary and full text content
        # summary = page.summary
        full_text = page.text

        return full_text

if selected=='Result':
    st.title('Result')

    if os.path.exists('static/save'):
        save_images_list()
        clear = st.button('Clear')
        if clear:
            shutil.rmtree('static/save')
            st.write('You can save new images now')
        st.divider()
        st.subheader('Know about the bird')
        st.write('number represent their order')
        choice = st.selectbox('Chose a number',options=[(choice+1) for choice in range(0,saved_images)])
        print(choice)
        result = st.button('predict')
        if result:
            if choice:
                file_paths = sorted([os.path.join('static/save', filename) for filename in os.listdir('static/save') if os.path.isfile(os.path.join('static/save', filename))])
                print(file_paths)
                img = img_preprocess(file_paths[choice-1])

                class_name,confidence = pred_label(img)
                print(class_name,confidence)
                
                if confidence < 0.9:
                    st.write("I'm sorry, but I'm not prepared for this picture.")
                    st.divider()
                    st.subheader('why ?')
                    st.markdown(
                    """ 
                    -  _It has to be the bird image._

                    - _blurry or unrecognizable image of a bird_

                    - _Might not have received training for this image._

                    """
                    )
                else:
                    full_text = get_bird_info(class_name)
                    st.write(f'According to the image you provided, the bird is most likely a **"{class_name}"**')    
                    st.divider()
                    st.header('Full Information')
                    st.image(img)
                    st.subheader(f'{class_name} [Confidence: {confidence*100:.4}%]')
                    st.write(full_text)

    else:
        st.subheader('No saved images found')
    # st.write(summary)
               
#============================================================================================================================
#About
if selected=='About':
    st.title('Data')
    #st.subheader('All data for this project was publicly sourced from:')
    col1,col2,col3=st.columns(3)
    col1.subheader('Source')
    col2.subheader('Description')
    col3.subheader('Link')
    with st.container():
        col1,col2,col3=st.columns(3)
        col1.write(':blue[Kaggle]')
        col2.write('BIRDS 525 SPECIES- IMAGE CLASSIFICATION')
        col3.write('https://www.kaggle.com/datasets/gpiosenka/100-bird-species')
    
    with st.container():
        col1,col2,col3=st.columns(3)
        #col1.image('cdc.png',width=150)
        col1.write(':blue[Google Drive]')
        col2.write('Actual dataset produced utilizing the aforementioned Kaggle datasets to train the model')
        col3.write('https://drive.google.com/file/d/1lSVE0gBQChGBvjVjNbf6ewDO7pJ7Q2zU/view?usp=drive_link')
    
    st.divider()
    
    st.title('Creator')
    with st.container():
        col1,col2=st.columns([0.7,0.3])
        col1.write('')
        col1.write('')
        col1.write('')
        col1.write('**Name:**    David Thapa')
        col1.write('**Education:**    Bachelor of Science in Computer Science and Information Technology')
        col1.write('**Contact:**    daybeat06@gmail.com or [linkedin](https://www.linkedin.com/in/dav1dtmagar/)')
        col1.write('**Thanks for stopping by!**')
        col2.image('static/creator/David_Thapa.jpg',width = 250)     
    st.divider()
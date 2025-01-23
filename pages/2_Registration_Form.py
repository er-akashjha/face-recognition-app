import streamlit as st
import cv2
from Home import face_rec
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

#st.set_page_config(page_title='Registration Formm',layout='centered')
st.subheader('Registration Form')

registrationForm = face_rec.RegistrationForm()

#step 1 : Collect person name and role
personName = st.text_input(label='Name',placeholder='First & Last Name')
role = st.selectbox(label='select your role',options=('Student','Teacher'))
#step 2: Collect facial embedding of that person
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24')
    rec_img,embedding = registrationForm.get_embedding(img)
    
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)

    return av.VideoFrame.from_ndarray(rec_img,format='bgr24')

webrtc_streamer(key='registration',video_frame_callback=video_callback_func,
                rtc_configuration={
        "iceServers": [{"urls": ["stun:bn-turn2.xirsys.com"]},{"username":"Ziq2HUZIVy4GSOgu-HjRW8jPTh6aBi8KMTFyica9LCSaDS3wUGkbAtV90d9KeCAkAAAAAGeSUpZhNGFrYXNoamhh","credential":"a2a41a10-d996-11ef-a0ce-0242ac140004","urls":["turn:bn-turn2.xirsys.com:80?transport=udp","turn:bn-turn2.xirsys.com:3478?transport=udp","turn:bn-turn2.xirsys.com:80?transport=tcp","turn:bn-turn2.xirsys.com:3478?transport=tcp","turns:bn-turn2.xirsys.com:443?transport=tcp","turns:bn-turn2.xirsys.com:5349?transport=tcp"]}]
    }
)
#step 3: Save the data in redis database

if st.button('Submit'):
    return_val = registrationForm.save_data_in_redis_db(personName,role)
    if return_val==True:
        st.success(f"{personName} Registered Successfully")
    else:
        st.error('Something Went Wrong')

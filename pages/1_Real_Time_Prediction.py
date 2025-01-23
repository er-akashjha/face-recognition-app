import time
import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av

#st.set_page_config(page_title='Real Time Attendance System',layout='centered')
st.subheader('Real Time Attendance System')

with st.spinner('Retriving Data from Redis DB....'):
    redis_face_db=face_rec.retrive_data(name='s2s:school')
    st.dataframe(redis_face_db)
st.success('Data retrive from DB')

# streamlit webrtc

#time
waitTime=30 #30 seconds wait
setTime = time.time()
realtimePred = face_rec.RealTimePred()

#callback function
def video_frame_callbacks(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24") #3d array to numpy array
    print('invoked face_prediction')
    pred_image = realtimePred.face_prediction(img,redis_face_db,'facial_features')
    timeNow = time.time()
    diffTime = timeNow-setTime

    if diffTime>=waitTime:
        realtimePred.saveLogs_redis()
        setTime= time.time()
        print("Data saved to Redis")

    return av.VideoFrame.from_ndarray(pred_image, format="bgr24")


webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callbacks,
rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
import cv2
import streamlit as st
import numpy

cap = cv2.VideoCapture(0)

st.title('Test')

frame_placeholder = st.empty()

stop_button_pressed = st.button("Stop")

while cap.isOpened() and not stop_button_pressed:

    ret,frame = cap.read()

    if not ret:
        st.write("Video capture has ended")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame,channels="RGB")

    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
        break

cap.release()
cv2.destroyAllWindows()

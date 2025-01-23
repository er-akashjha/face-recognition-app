import streamlit as st
from Home import face_rec
#st.set_page_config(page_title='Report',layout='wide')
st.subheader('Report')

name='attendance:logs'

def load_logs(name,end=-1):
    logs_list=face_rec.r.lrange(name,start=0,end=end)
    return logs_list

tab1,tab2 =st.tabs(['Registered Data','Logs'])

with tab2:
    #if st.button('Refresh Logs'):
    st.write(load_logs(name=name))

with tab1:
   # if st.button('Refresh Data'):
    with st.spinner('loading'):
        redis_face_db=face_rec.retrive_data(name='s2s:school')
        st.dataframe(redis_face_db)
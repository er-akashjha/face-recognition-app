import numpy as np
import pandas as pd
import cv2
import redis
import time
from datetime import datetime
import os

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# connect to redis client
hostname = 'redis-19338.crce179.ap-south-1-1.ec2.redns.redis-cloud.com'
portnumber = 19338
password = '5ZvefspUjdWfm9jUPs4FUsSw80gBiETy'

r = redis.StrictRedis(host=hostname,
                      port = portnumber,
                      password=password)

#Retrieve data
#name= 's2s:school'
def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x:x.decode(),index))
    retrive_series.index = index
    retrive_series_dataframe = retrive_series.to_frame().reset_index()
    retrive_series_dataframe.columns=['name_role','facial_features']
    retrive_series_dataframe[['Name','Role']]=retrive_series_dataframe['name_role'].apply(lambda x:x.split('@')).apply(pd.Series)
    return retrive_series_dataframe[['Name','Role','facial_features']]

# configure face analysis
faceapp = FaceAnalysis('buffalo_l','insightface_model')
faceapp.prepare(ctx_id=0,det_size=(640,640), det_thresh = 0.5)

# ML Search Algorithm
def ml_search_algorithm(dataframe,feature_column, test_vector,thresh=0.5):
    """
    cosine similarity base search algorith
    """
    #step-1 : take the dataframe (collection of data)
    dataframe = dataframe.copy()
    
    #step-2 : Index face embedding from the dataframe and convert into array
    x_list = dataframe[feature_column].tolist()
    x=np.asarray(x_list)
    
    #step-3 : Cal. Cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr
    #step-4 : filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter)>0:
        #step-5 : get the person name
        data_filter.reset_index(drop=True,inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name,person_role = data_filter.loc[argmax][['Name','Role']]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
    #current_time = datetime.datetime.now()
    return person_name, person_role



## We need to save logs for every 1 min
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[],role=[],current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[],role=[],current_time=[])

    def face_prediction(self,test_image,dataframe,feature_column,thresh=0.5):
            #step-0 : find the time
            current_time = str(datetime.now())
            
            #step-1 : take the test image and apply to insight face
            results = faceapp.get(test_image)
            test_copy = test_image.copy()
            #step-2: use for loop and extract each embedding and pass to ml_search_algorithm
            
            for res in results:
                x1,y1,x2,y2 = res['bbox'].astype(int)
                embeddings = res['embedding']
                print("Goint to enter in search algo")
                person_name, person_role = ml_search_algorithm(dataframe,feature_column,embeddings)
                print("Search algo completed found person as")
                print(person_name)
                if person_name=='Unknown':
                    color=(0,0,255)
                else:
                    color=(0,255,0)
                
                print(person_name , person_role, current_time)
                cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
                text_gen = person_name
                cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)
                print('rectangle and name should be printed')
                cv2.putText(test_copy,current_time,(x2,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)
                print('Time should be printed')
                #save info in logs dict
                self.logs['name'].append(person_name)
                self.logs['role'].append(person_role)
                self.logs['current_time'].append(current_time)
            
            return test_copy

    def saveLogs_redis(self):
        # step:1-> create a logs dataframe
        dataframe = pd.DataFrame(self.logs)
        #step:2-> drop the duplicate information(distinct name)
        dataframe.drop_duplicates('name',inplace=True)
        #step:#-> Push data to redis databse (list)
        #encode the data
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data=[]
        for name,role,ctime in zip(name_list,role_list,ctime_list):
            if name!='Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)
            
            if len(encoded_data)>0:
                r.lpush('attendance:logs',*encoded_data)

            self.reset_dict()

    
#### Registration Form
class RegistrationForm:
    def __init__(self):
        self.sample=0
    def reset(self):
        self.sample=0

    def get_embedding(self,frame):
        embeddings=None
        results = faceapp.get(frame,max_num=1)
        for res in results:
            self.sample+=1
            x1,y1,x2,y2 = res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            text = f"Samples collected = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            cv2.putText(frame,"S2S",(10,10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.6,(255,0,0),3)
            embeddings = res['embedding']
            #face_embeddings.append(embeddings)
        return frame,embeddings
    
    def save_data_in_redis_db(self,name,role,user_id):
        if name is not None:
            if name.strip()!='':
                key=f'{name}@{role}'
            else:
                return 'Name is incorrect'
        else:
            return 'No Name'
        
        if f"face_embedding_{user_id}.txt" not in os.listdir():
            return 'file_false'

        file_name = f"face_embedding_{user_id}.txt"
        x_array = np.loadtxt(file_name,dtype=np.float32)
        
        received_samples = int(x_array.size/512)
        x_array=x_array.reshape(received_samples,512)
        x_array=np.asarray(x_array)

        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes=x_mean.tobytes()

        r.hset(name='s2s:school',key=key,value=x_mean_bytes)
        os.remove(f"face_embedding_{user_id}.txt")
        self.reset()

        return True

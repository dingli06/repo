import requests
from base64 import b64encode
import cv2 as cv
import numpy as np
from server import Face
face=Face()
face_info=face.vec

def byte2b64(byte):
    return b64encode(byte).decode()

def byte2arr(byte):
    img_np=np.frombuffer(byte,dtype=np.uint8)
    img=cv.imdecode(img_np,cv.IMREAD_COLOR)
    return img

def read_records():
    return eval(open("records.txt",encoding="utf-8").read())

def write_records(key_id,key_name,key_sim,time):
    records=read_records()
    records.append([key_id,key_name,key_sim,time])
    with open("records.txt","w",encoding="utf-8") as w:
        w.write(str(records))
        w.close()

def write_base(base):
    w=open("face.base","w",encoding="utf-8")
    w.write(str(base))

def read_base():
    return eval(open("face.base","r",encoding="utf-8").read())

def arr2byte(arr):
    data = cv.imencode('.png', arr)[1]
    img_byte=data.tobytes()
    return img_byte

def save_img2base(path,img_byte):
    w=open(path,"wb")
    w.write(img_byte)

def modify_base(base,id_,name,vec,path):
    base[id_]={"name":name,"vec":vec,"path":path}
    write_base(base)


def splitface(image):
    data_path = 'haarcascade_frontalface_default.xml'
    classfier = cv.CascadeClassifier(data_path)
    faceRects = classfier.detectMultiScale(image, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    x, y, w, h = faceRects[0]
    face = image[y - 10: y + h + 10, x - 10: x + w + 10].copy()
    cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),4)
    face = cv.resize(face,(256,256))
    return image,face

'''
def face_info(b64):
    res=requests.post("http://127.0.0.1:8000/face",json={"img":b64}).json()
    return res
'''

def get_sim(v1,v2):
    return np.dot(v1,v2)


def match(b64,base,topk=0):
    vec=face_info(b64)["vec"]
    res=[[kid,kv["name"],kv["path"],get_sim(vec[0],kv["vec"][0])] for kid,kv in base.items()]
    if topk:
        return sorted(res,key=lambda x:x[-1],reverse=True)[:topk]
    else:
        return max(res,key=lambda x:x[-1])

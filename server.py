from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import numpy as np
from flask import Flask,request
from base64 import b64decode
import json


class Face:
    def __init__(self):
        self.model=rts_face_recognition_func = pipeline(Tasks.face_recognition, 'damo/cv_ir_face-recognition-ood_rts')

    def vec(self,b64):
        f=open("source.png","wb")
        f.write(b64decode(b64))
        f.close()
        result = self.model("source.png")
        emb = result[OutputKeys.IMG_EMBEDDING]
        score = result[OutputKeys.SCORES][0][0]
        return {"vec":emb.tolist(),"score":score}

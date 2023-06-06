import streamlit as st
from utils import *
import cv2 as cv
from datetime import datetime

st.set_page_config("人脸数据管理系统",layout="wide")

sb=st.sidebar

with sb:
    st.subheader("人脸管理系统")
    st.image("Im7lZjxeLhg.jpg")


t1,t2,t3=st.tabs(["人脸注册","人脸检索","人脸后台记录"])



with t1:
    option=st.selectbox("选择方式",["设备录入","照片上传"])
    lr,eval_lr=st.columns([3,2])
    with lr:
        if option=="设备录入":
            cm=st.camera_input("开始录入")
        else:
            cm=st.file_uploader("上传照片")
    st.write(cm)
    with eval_lr:        
        if cm:
            try:
                cb_byte=cm.getvalue()
            except:
                cb_byte=cm
                
            cb_b64=byte2b64(cb_byte)
            face_res=face_info(cb_b64)
            
            st.write("人脸质量评分:{:.2f}".format(face_res["score"]))
            cb_arr=byte2arr(cb_byte)
            cb_arr_detect,face=splitface(cb_arr)
            st.image(

                cb_arr_detect[:,:,::-1]
            )

            lr_name=st.text_input("录入人:")
            lr_id=st.text_input("录入ID号:",value="最好为ID后8位")
            button_state=st.button("提交",use_container_width=True)
            if button_state:
                if not lr_name:
                    st.info("录入人不能为空")
                if not lr_id:
                    st.info("录入ID不能为空")
                if lr_name and lr_id:
                    base=read_base()
                    if lr_id not in base:        
                        path=f"baseimgs/{lr_id}.png"
                        save_img2base(path,arr2byte(face) )
                        st.success("录入成功")
                    else:
                        st.warning("录入失败,此id已经存在")
                    modify_base(base,lr_id,lr_name,face_res["vec"],path)

        else:
            st.subheader("录入说明:")
            st.info("整体要求: 免冠(不戴帽子) ")
            st.info("背景要求: 采用白色、蓝色等纯色背景 (只要不和皮肤颜色相同即可)")
            st.info("化妆要求: 拍照时不得有影响真实面貌的化妆色彩，包括头发的染色和睫毛等.")
            st.info("光线要求:拍照时需要有亮度合适的光线，不应该存在过暗、过亮或阴阳脸等现象。")
    
    with t2:
        option=st.selectbox("选择方式",["设备录入","照片上传"],key="js")
        js_type=st.radio("",["单人检索","相似匹配"],horizontal=True)
        if js_type=="相似匹配":
            topk_num=st.number_input("TopK数",min_value=5,max_value=10)
        js,eval_js=st.columns([2,2])
        

        with js:
            if option=="设备录入":
                cm=st.camera_input("开始录入",key="jslr")
            else:
                cm=st.file_uploader("上传照片",key="jssc")

        with eval_js:
            if cm:
                cb_byte=cm.getvalue()
                cb_b64=byte2b64(cb_byte)
                base=read_base()
                if js_type=="单人检索":
                    st.info("匹配成功")
                    key_id,key_name,key_path,key_sim=match(cb_b64,base)
                    if key_sim>0.7:
                        write_records(key_id,key_name,key_sim,datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    st.markdown("ID:{}\tName:{}\t相似度:{:.2f}".format(key_id,key_name,key_sim))
                    st.image(key_path,width=100)
                else:
                    for r in match(cb_b64,base,topk=topk_num):
                        key_id,key_name,key_path,key_sim=r
                        st.markdown("**ID:{} 相似度:{:.2f}**".format(key_id,key_sim))
                        #with elements(key_id):
                        #mui.Card(mui.List(mui.ListItem(mui.ListItemText(primary=f"ID:{key_id}",secondary="相似度:{:.2f}".format(key_sim)))))
                        st.image(key_path,width=80)
    with t3:
        records=read_records()
        if records:
            for r in records:
                st.markdown("**id:{} time :{}**".format(r[0],r[3]))
        else:
            st.subheader("暂无记录")

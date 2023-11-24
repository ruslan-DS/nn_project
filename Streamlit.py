import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import streamlit as st
import plotly_express as px
from plotly.tools import mpl_to_plotly
import io
import imageio
import requests
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet34, resnet18
# from torchvision import io
import torch.nn as nn
import time



model = resnet34()
model.fc = nn.Linear(512, 11)
device = 'cpu'
model.load_state_dict(torch.load('best_params_resnet34.pt',  map_location=torch.device('cpu')))
func_preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])
def preprocess(image: str):
    return func_preprocess(image)
def get_prediction(image) -> str:
    dict_classes = {0: 'dew',
        1: 'fogsmog',
        2: 'frost',
        3: 'glaze',
        4: 'hail',
        5: 'lightning',
        6: 'rain',
        7: 'rainbow',
        8: 'rime',
        9: 'sandstorm',
        10: 'snow'}
  
    image = preprocess(image)
    device = 'cpu'
    model.to(device)
    model.eval()
    classes = (torch.argmax(model(image.unsqueeze(0).to(device)), dim=1)).item()
    
    return f'Модель предсказала: {dict_classes[classes]}'

# model2 = resnet18()
# model.fc = nn.Linear(INPUT_SIZE, 200)
# model.load_state_dict(torch.load('ВЕСАААААААА'))
# func_preprocess2 = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor()
# ])
# def preprocess(image2: str):
#     return func_preprocess2(image2)
# def get_prediction(image) -> str:
#     dict_classes = {}
  
#     image2 = preprocess(image2)
#     device = 'cpu' 
#     model2.to(device)
#     model2.eval()
#     classes2 = (torch.argmax(model(image.unsqueeze(0).to(device)), dim=1)).item()
    
#     return f'Модель предсказала: {dict_classes[classes2]}'


st.title('Проект • Введение в нейронные сети')

st.sidebar.header('Выберите страницу')
page = st.sidebar.selectbox("Выберите страницу", ["Главная", "Времена года", "Кстати, о птичках", "Итоги"])

if page == "Главная":
        st.header('Выполнила команда "DenseNet":')
        st.subheader('🐱Руслан')
        st.subheader('🐱Тата')

        st.header(" 🌟 " * 10)

        st.header('Наши датасеты')
        st.subheader('*Задача №1*: Классификация изображений природы по временам года (обучение на основе датасета *Weather Image Recognition*)')

        st.subheader('*Задача №2*: Определение вида птички по фотографии.')


elif page == "Времена года":

        image_url = st.text_input("Введите URL изображения погоды")
        start_time = time.time()

        if image_url:
    # Загрузка изображения по ссылке
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
                st.subheader('Ваше фото')
                st.image(image)
                st.subheader('Предсказание модели')
                st.write(get_prediction(image))
                st.subheader('Время предсказания')
                st.subheader(time.time() - start_time)
                st.header('🎈' * 10)

 
                        
                 
elif page == "Кстати, о птичках":
        st.subheader("")
        st.markdown("")
        image_url3 = st.text_input("Введите URL изображения вашей птички")

        if image_url3:
    # Загрузка изображения по ссылке
                response3 = requests.get(image_url3)
                image3 = Image.open(io.BytesIO(response3.content))
                st.subheader('Фото вашей птички')
                st.image(image3)
                st.subheader("Мы узнали, что это за птичка")
                #НАШ КОД
elif page == "Итоги":
        st.subheader('Результаты и выводы')
        #РАССКАЗ О ТОМ, КАК НАМ БЫЛО ТЯЖЕЛО, НО МЫ СПРАВИЛИСЬ







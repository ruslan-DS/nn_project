import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import streamlit as st
import plotly_express as px
from plotly.tools import mpl_to_plotly
import io
import imageio
import requests
from PIL import Image

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
        st.subheader("")
        st.markdown("")
        image_url = st.text_input("Введите URL изображения погоды")

        if image_url:
    # Загрузка изображения по ссылке
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
                st.subheader('Ваше фото')
                st.image(image)

elif page == "Кстати, о птичках":
        st.subheader("")
        st.markdown("")
        image_url2 = st.text_input("Введите URL изображения вашей птички")

        if image_url2:
    # Загрузка изображения по ссылке
                response2 = requests.get(image_url2)
                image2 = Image.open(io.BytesIO(response2.content))
                st.subheader('Фото вашей птички')
                st.image(image2)
                st.subheader("Мы узнали, что это за птичка")
                #НАШ КОД
elif page == "Итоги":
        st.subheader('Результаты и выводы')
        #РАССКАЗ О ТОМ, КАК НАМ БЫЛО ТЯЖЕЛО, НО МЫ СПРАВИЛИСЬ







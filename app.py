import streamlit as st
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import requests

col1, mid, col2 = st.columns([1,15,25])
with col1:
    st.image('data/logo.jpeg', width=150)
with col2:
    st.title('Clothing Store Recommender')


with st.expander("See explanation"):
     st.write("""
         This Project (web app) is designed to recommend cloth store based on the user input.
         Attributes taken for decision making - Age, Gender, Garment Preference,
         Branded(or Not), Price and Rating.
         * This Project is Scalable
         * Modular in approach
         * Code is Easy to read and Understand
     """)

st.subheader('')
st.subheader('')

video_file = open('data/videoplayback.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

st.subheader('')
st.subheader('Dataset-: ')

path1='data/ClothFinal.csv'
df = pd.read_csv(path1)
st.dataframe(df)

path='data/Store.csv'
new_df = pd.read_csv(path)
new_df = new_df.drop(new_df.columns[[0]],axis = 1)

with st.expander("See explanation"):
     st.write("""
         In this dataset is self created and i have taken 25 stores for this
         ( can be expanded acc to the future requirements)
         * For Age- 
            - 3-10 ( Children )
            - 10-20 (Teenage )
            - 20-40 ( Adult )
            - 40-60 ( Senior )
        * Garment Preference-  have included some preferences...
            - like -> Shirt, TShirt, Casual, Formal, Top, Innerwear, ChildWear, Kurta,
                     Saree, Suit,etc.
        * Branded- ( as Branded or Non Branded )
        * Ratings- ( only from 3 to 5 ... beoz don't want to recommend stores with less than 3 rating )
     """)


st.subheader('')
st.subheader('')
st.subheader('Enter your preferences-: ')

age = st.radio(
     "Enter Age-: ",
     ('Children','Teenage','Adult','Senior'))

gender = st.radio(
     "Enter Gender-: ",
     ('Male', 'Female'))

garment = st.text_input('Garment')
st.write('Garment Type-: ', garment)

branded = st.radio(
     "Enter Branded or not-: ",
     ('Branded', 'NonBranded'))

price = st.text_input('Price')
st.write('Enter Price-: ', price)

rating = st.radio(
     "Enter ratings-: ",
     ('3', '4', '5'))


tag= age +' '+ gender +' ' + garment +' ' + branded  +' ' + price +' '+ rating
tag=tag.lower()

st.write(tag)

def recommend(tag,new_df):
    new = new_df
    new.loc[len(new.index)] = [0,'', tag]
    cv = CountVectorizer(max_features=500, stop_words='english')
    vector = cv.fit_transform(new['tags']).toarray()
    similarity = cosine_similarity(vector)

    index = new[new['tags'] == tag].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_store_names = []
    for i in distances[1:4]:
        recommended_store_names.append(new.iloc[i[0]].Store)
    del new
    return recommended_store_names



if st.button('Show Recommendation'):
    recommended_store_names = recommend(tag,new_df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('')
        st.text('1 - ')
        st.text(recommended_store_names[0])
    with col2:
        st.subheader('')
        st.text('2 - ')
        st.text(recommended_store_names[1])
    with col3:
        st.subheader('')
        st.text('3 - ')
        st.text(recommended_store_names[2])




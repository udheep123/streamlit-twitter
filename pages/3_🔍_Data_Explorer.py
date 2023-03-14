import streamlit as st
import pandas as pd

st.set_page_config(page_title="Tweets Explorer", page_icon="🌍")

st.markdown("<h1 style='text-align: center; color: black;'>Tweets Data Explorer</h1>", unsafe_allow_html=True)

df = st.session_state["tw_data1"]

# st.markdown("<b> <h2 style='text-align: left;'>Tweets Viewer</h2> </b>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    option1 = st.selectbox(
    "View Tweets based on",
    ("Sentiment", "Topic"), key="filter1")
view_txt = "Which "+option1+" would you like to see?"

if option1=='Topic':
    list_disp = df['topic_name_final'].unique().tolist()
    col_disp = 'topic_name_final'
if option1=='Sentiment':
    list_disp = df['sentiment_tweetnlp'].unique().tolist()
    col_disp = 'sentiment_tweetnlp'
else:
    pass

with col2:
    option2 = st.selectbox(
    view_txt,
    (list_disp),key="filter2")

if ('option1' in locals()) & ('option2' in locals()):
    st.write(df[df[col_disp]==option2][['date','renderedContent']].sort_values('date', ascending=False))

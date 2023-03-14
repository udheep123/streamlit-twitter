import streamlit as st
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Tweets Sentiment", page_icon="📊")

st.markdown("<h1 style='text-align: center; color: black;'>Tweets Sentiment</h1>", unsafe_allow_html=True)

df = st.session_state["tw_data1"]
wordcloud = st.session_state["wordcloud1"]
wordcloud_ps = st.session_state["wordcloud_ps1"]
# st.session_state["wordcloud_nu1"] = wordcloud_nu
wordcloud_ne = st.session_state["wordcloud_ne1"]

col1, col2 = st.columns([4.5,1])

# with col1:
#     st.markdown("<b> <h2 style='text-align: left;'>Hashtag Trend of </h2> </b>", unsafe_allow_html=True)
# #             st.write("(Hover on for more details)")
# with col2:  
#     tweet_type = st.selectbox("", ("All Tweets","Positive Tweets","Negative Tweets"))

with col2:  
    tweet_type = st.radio("HashTag Trend of",('All Tweets', 'Positive', 'Negative'))

with col1:  
    if tweet_type=='All Tweets':
        plt.subplots(figsize = (21,7))
        plt.imshow(st.session_state["wordcloud1"]) # image show
        plt.axis('off')
        st.pyplot(plt)
    if tweet_type=='Positive':
        plt.subplots(figsize = (21,7))
        plt.imshow(st.session_state["wordcloud_ps1"]) # image show
        plt.axis('off')
        st.pyplot(plt)
    # if tweet_type=='Neutral Tweets':
    #     plt.subplots(figsize = (21,7))
    #     plt.imshow(st.session_state["wordcloud_nu1"]) # image show
    #     plt.axis('off')
    #     st.pyplot(plt)
    if tweet_type=='Negative':
        plt.subplots(figsize = (21,7))
        plt.imshow(st.session_state["wordcloud_ne1"]) # image show
        plt.axis('off')
        st.pyplot(plt)      


st.markdown("---")
# start = time.time()
fig1 = px.pie(df, values='number', names='sentiment_tweetnlp', color_discrete_sequence=["#26828E", "#3F4889"])
fig1.update_layout(legend_title="Sentiment")

# fig1.update_layout(margin=dict(t=0, b=0, l=0, r=0))
#line chart for sentiments
df_sentiment_week_agg = pd.DataFrame(df.groupby(['year_week','sentiment_tweetnlp'])['number'].count())
df_sentiment_week_agg.reset_index(inplace=True)
df_sentiment_week_agg = pd.crosstab(index=df_sentiment_week_agg['year_week'], columns=[df_sentiment_week_agg['sentiment_tweetnlp']], values=df_sentiment_week_agg.number, aggfunc=sum)
df_sentiment_week_agg.reset_index(inplace=True,drop=False)
df_sentiment_week_agg.fillna(0,inplace=True)
fig2 = px.line(df_sentiment_week_agg, x='year_week', y=["negative","positive"],width=700, height=300,labels={'value':'# Tweets','year_week':'Year Week','variable':'Legends'}, color_discrete_sequence=["#3F4889", "#26828E"])
fig2.update_layout(xaxis_title=None)
#         st.write(str(df.shape[0])+" rows of data extracted for 4th time "+ tw_name)
col1,col2 = st.columns([2,1])
with col1:
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    st.plotly_chart(fig1, use_container_width=True)



# st.write("translation complete and time taken is ",(time.time()-start))

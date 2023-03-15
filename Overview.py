import streamlit as st
import pandas as pd
from datetime import date
from datetime import timedelta
# import statsmodels.api as sm
import numpy as np 
import plotly.express as px
from numerize import numerize
import base64
import snscrape.modules.twitter as sntwitter
from streamlit_metrics import metric, metric_row
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn import preprocessing
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# import tweetnlp
#from pandarallel import pandarallel
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime,timezone
from bertopic import BERTopic
from umap import UMAP
import urllib3, socket
from urllib3.connection import HTTPConnection
import multiprocessing
import time
from lang_trans_basic import lang_chg
from googletrans import Translator
import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import joblib
from flair.models import TextClassifier
from flair.data import Sentence

HTTPConnection.default_socket_options = ( 
    HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000), #1MB in byte
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 1000000)
])

# Define custom CSS to remove empty space


#def lang_chg(tweet):
#    from googletrans import Translator
#    translator = Translator()
#    return translator.translate(tweet,dest= "en").text

st.set_page_config(layout="wide",page_title='Twitter Profiler',page_icon = "🌏")

st.markdown("<h1 style='text-align: center; color: black;'>Twitter Profiler</h1>", unsafe_allow_html=True)
# st.title("Twitter Profiler")

if "userScraper1" not in st.session_state:
    st.session_state["userScraper1"] = ""
if "tw_data1" not in st.session_state:
    st.session_state["tw_data1"] = ""

col1, col2 = st.columns([3,2])
with col1:
    if "tw_name1" not in st.session_state:
        st.session_state["tw_name1"] = ""
    tw_name = st.text_input("",key='name',placeholder="Please enter the Twitter Handle")
with col2:
    if "analysis_months1" not in st.session_state:
        st.session_state["analysis_months1"] = ""
    analysis_months = st.text_input("", key='months', placeholder="# Months to Analyze")

# st.write(tw_name,analysis_months)  
submit = st.button("Submit")
if submit:
#     st.write(tw_name,analysis_months)  
    st.session_state["tw_name1"] = tw_name
    st.session_state["analysis_months1"] = analysis_months 
    userScraper = sntwitter.TwitterUserScraper(tw_name).entity
    st.session_state["userScraper1"] = userScraper 

# tw_name = my_input
# analysis_months = 6
st.markdown("---")

if st.session_state["tw_name1"] is not None:
# #display user display name and image
    userScraper = st.session_state["userScraper1"]
# if userScraper:
    try:
        col1, col2 = st.columns([3,1])
        with col1:
            st.title(userScraper.displayname) 
            st.subheader(userScraper.rawDescription)
        with col2:
            st.image(userScraper.profileImageUrl.replace("_normal",""), width = 220)
        # Twitter Profile summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("#Followers", format(userScraper.followersCount, ','))
        col2.metric("#YearsOnTwitter", format(round((datetime.now(timezone.utc)-userScraper.created).days/365.25), ','))
        col3.metric("#Statuses", format(userScraper.statusesCount, ','))
        col4.metric("#Favourites", format(userScraper.favouritesCount, ','))
        st.markdown("---")

    except:
        st.title("Twitter Handle Doesn't exist") 

if submit:
    with st.spinner('Scraping Account Data...'):
        query = "(from:@"+tw_name+") include:nativeretweets until:"+str(date.today()+timedelta(days=1))+" since:"+str(date.today()-timedelta(days=round(int(analysis_months)*30.5)))
        tweets = []
        limit = 1000

        for tweet in sntwitter.TwitterSearchScraper(query).get_items():

            if len(tweets) == limit:
                break
            else:
                tweets.append([tweet.url, tweet.date, tweet.rawContent, tweet.renderedContent,tweet.id,tweet.user.username,tweet.user.id,
                              tweet.user.displayname,tweet.user.rawDescription,tweet.user.renderedDescription,tweet.user.descriptionLinks,
                              tweet.user.verified,tweet.user.created,tweet.user.followersCount,tweet.user.friendsCount,
                              tweet.user.statusesCount,tweet.user.favouritesCount,tweet.user.listedCount,tweet.user.mediaCount,
                              tweet.user.location,tweet.user.protected,tweet.user.link,tweet.user.profileImageUrl,
                              tweet.user.profileBannerUrl,tweet.user.label,tweet.replyCount,tweet.retweetCount,tweet.likeCount
                              ,tweet.quoteCount,tweet.conversationId,tweet.lang,tweet.source,tweet.sourceUrl,tweet.sourceLabel
                              ,tweet.links,tweet.media,tweet.retweetedTweet,tweet.quotedTweet,tweet.inReplyToTweetId
                              ,tweet.inReplyToUser,tweet.mentionedUsers,tweet.coordinates,tweet.place,tweet.hashtags
                              ,tweet.cashtags,tweet.card,tweet.viewCount,tweet.vibe])

        df = pd.DataFrame(tweets, columns=['url','date', 'rawContent','renderedContent','id','username','id','displayname', 'rawDescription','renderedDescription','descriptionLinks','verified','created','followersCount','friendsCount'
                                          ,'statusesCount','favouritesCount','listedCount','mediaCount','location','protected','link'
                                          ,'profileImageUrl','profileBannerUrl','label','replyCount','retweetCount','likeCount'
                                          ,'quoteCount','conversationId','lang','source','sourceUrl','sourceLabel','links'
                                          ,'media','retweetedTweet','quotedTweet','inReplyToTweetId','inReplyToUser','mentionedUsers'
                                          ,'coordinates','place','hashtags','cashtags','card','viewCount','vibe'])

        df=df[(df['retweetedTweet'].isna()) & (df['quotedTweet'].isna())]
        df.reset_index(inplace=True, drop=True)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        df['year_week'] = df['year'].astype(str)+"W"+df['week'].astype(str).str.pad(width=2,side='left',fillchar='0')
        df['year_week'] = np.where((df['month']==1) & (df['week']>50),df['year'].astype(str)+"W01",df['year_week'])
        st.session_state["tw_data1"] = df

try:
    df = st.session_state["tw_data1"]
    df['renderedContent'] = df['rawContent']
    df_week_agg = df.groupby('year_week').agg({'id':'size','replyCount':'sum','retweetCount':'sum','likeCount':'sum','viewCount':'sum', 'date':'min'})
    df_week_agg['replyCountAvg'] = df_week_agg['replyCount']/df_week_agg['id']
    df_week_agg['retweetCountAvg'] = df_week_agg['retweetCount']/df_week_agg['id']
    df_week_agg['likeCountAvg'] = df_week_agg['likeCount']/df_week_agg['id']
    df_week_agg['viewCountAvg'] = df_week_agg['viewCount']/df_week_agg['id']
    df_week_agg.reset_index(inplace=True)

    fig = make_subplots(rows=1, cols=4,subplot_titles=("Tweets", "Retweets", "Likes", "Views"))

    fig.add_trace(go.Scatter(x=df_week_agg.date, y=df_week_agg.id, line_color='#3F4889'),row=1, col=1)
    fig.add_trace(go.Scatter(x=df_week_agg.date, y=df_week_agg.retweetCountAvg, line_color='#481668'),row=1, col=2)
    fig.add_trace(go.Scatter(x=df_week_agg.date, y=df_week_agg.likeCountAvg, line_color='#26828E'),row=1, col=3)
    fig.add_trace(go.Scatter(x=df_week_agg.date, y=df_week_agg.viewCountAvg, line_color='#9FDAC5'),row=1, col=4)
    fig.update_layout(height=300, width=600, showlegend=False, title_text="Account Activity Over Time",title_font=dict(size=25))
    fig.update_xaxes(showticklabels=True)
    
    st.plotly_chart(fig, use_container_width=True)    
    # st.markdown("---")

except:
    st.title("Data Doesn't exist")

if submit:
    with st.spinner('Processing Tweets for Sentiment...'):
        start = time.time()
        df['rawContent'] = [' '.join([ word for word in tweet.split() if not word.startswith('https') ]) for tweet in df['rawContent'].tolist()]    
        df['rawContent'] = [' '.join([ word for word in tweet.split() if not word.startswith('http') ]) for tweet in df['rawContent'].tolist()]
        df = df[df['rawContent']!=""]
        df = df[df['rawContent'].notnull()]
        df.reset_index(inplace=True, drop=True)

        df['rawContent'] = [re.sub("""[#'.…"“”|:,#!-?’&@()♦️]""","",col) for col in df['rawContent'].tolist()]

        df_nonen = df[df['lang']!="en"][['rawContent']]

        # translator = Translator()
        # df_nonen['cleaned_tweets_trans'] = df_nonen['rawContent'].apply(lambda x:translator.translate(x,dest= "en").text)
        df_nonen['cleaned_tweets_trans'] = joblib.Parallel(n_jobs=-1)(joblib.delayed(lang_chg)(i) for i in df_nonen['rawContent'].tolist())

        df = df.join(df_nonen[['cleaned_tweets_trans']])
        # df['cleaned_tweets'] = df.rawContent.combine_first(df.cleaned_tweets_trans)
        df['cleaned_tweets'] = np.where(df['lang']=="en", df['rawContent'], df['cleaned_tweets_trans'])
        lemmatizer = WordNetLemmatizer()
        df['cleaned_tweets']  = [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in df['cleaned_tweets']]
        stopwords = nltk.corpus.stopwords.words('english')
        df['cleaned_tweets_tm'] = df['cleaned_tweets'].apply(lambda x: ' '.join([w for w in x.split() if w.lower() not in stopwords]))
        df = df[df['cleaned_tweets_tm'].notnull()]
        df.reset_index(inplace=True, drop=True)

        st.write("Execution complete and time taken is ",(time.time()-start))
        #Sentiment Analysis
        # @st.cache_resource()
        # def load_model():
        #     model = tweetnlp.load_model('sentiment')
        #     return model
        # model_tweetnlp = load_model()

        @st.cache_resource()
        def load_model():
            sia = TextClassifier.load('en-sentiment')
            return sia
        model_flair = load_model()

        def flair_prediction(x):
            sentence = Sentence(x)
            model_flair.predict(sentence)
            return str(sentence.labels[0]).split('→')[-1]

        start = time.time()
        # df['sentiment_tweetnlp'] = df['cleaned_tweets'].apply(lambda x: model_tweetnlp.sentiment(str(x))['label'])
        df["sentiment"] = df["cleaned_tweets"].apply(flair_prediction)
        df["sentiment_tweetnlp"] = df['sentiment'].apply(lambda x: x.split(" ")[1])
        df["sentiment_tweetnlp"] = df["sentiment_tweetnlp"].str.lower() 
        df["sentiment_score"] = df['sentiment'].apply(lambda x: x.split(" ")[2])
        
        
        st.write("sentiment using apply complete and time taken is ",(time.time()-start))

        df['number'] = 1
        
        hashtag_ls = ' '.join([' '.join(col) for col in df.hashtags.tolist() if col!=None])
        wordcloud = WordCloud (background_color = 'white',width = 1200,height = 400,collocations=False, colormap='viridis').generate(hashtag_ls)
        hashtag_ls_ps = ' '.join([' '.join(col) for col in df[df['sentiment_tweetnlp']=='positive'].hashtags.tolist() if col!=None])
        wordcloud_ps = WordCloud (background_color = 'white',width = 1200,height = 400,collocations=False, colormap='Greens').generate(hashtag_ls_ps)
        # hashtag_ls_nu = ' '.join([' '.join(col) for col in df[df['sentiment_tweetnlp']=='neutral'].hashtags.tolist() if col!=None])
        # wordcloud_nu = WordCloud (background_color = 'white',width = 1200,height = 400,collocations=False,colormap='Paired').generate(hashtag_ls_nu)
        hashtag_ls_ne = ' '.join([' '.join(col) for col in df[df['sentiment_tweetnlp']=='negative'].hashtags.tolist() if col!=None])
        wordcloud_ne = WordCloud (background_color = 'white',width = 1200,height = 400,collocations=False,colormap='Reds').generate(hashtag_ls_ne)

        # end = time.time()
        st.session_state["tw_data1"] = df
        st.session_state["wordcloud1"] = wordcloud
        st.session_state["wordcloud_ps1"] = wordcloud_ps
        # st.session_state["wordcloud_nu1"] = wordcloud_nu
        st.session_state["wordcloud_ne1"] = wordcloud_ne
        # st.write("translation complete and time taken is ",(end-start))

# try:
#     df = st.session_state["tw_data1"]
#     start = time.time()
#     fig1 = px.pie(df, values='number', names='sentiment_tweetnlp')
#     fig1.update_layout(legend_title="Sentiment")
#     # fig1.update_layout(margin=dict(t=0, b=0, l=0, r=0))


#     #line chart for sentiments
#     df_sentiment_week_agg = pd.DataFrame(df.groupby(['year_week','sentiment_tweetnlp'])['number'].count())
#     df_sentiment_week_agg.reset_index(inplace=True)
#     df_sentiment_week_agg = pd.crosstab(index=df_sentiment_week_agg['year_week'], columns=[df_sentiment_week_agg['sentiment_tweetnlp']], values=df_sentiment_week_agg.number, aggfunc=sum)
#     df_sentiment_week_agg.reset_index(inplace=True,drop=False)
#     df_sentiment_week_agg.fillna(0,inplace=True)
#     fig2 = px.line(df_sentiment_week_agg, x='year_week', y=["negative","positive"],width=700, height=300,labels={'value':'# Tweets','year_week':'Year Week','variable':'Legends'})
#     fig2.update_layout(xaxis_title=None, title_text="Tweets Sentiment Analysis",title_font=dict(size=25))
# #         st.write(str(df.shape[0])+" rows of data extracted for 4th time "+ tw_name)
#     col1,col2 = st.columns([2,1])
#     with col1:
#         st.plotly_chart(fig2, use_container_width=True)
#     with col2:
#         st.plotly_chart(fig1, use_container_width=True)
#     st.markdown("---")

#     col1, col2, col3 = st.columns([0.75,0.4,1])

#     with col1:
#         st.markdown("<b> <h2 style='text-align: left;'>Hashtag Trend of </h2> </b>", unsafe_allow_html=True)
# #             st.write("(Hover on for more details)")
#     with col2:  
#         tweet_type = st.selectbox("", ("All Tweets","Positive Tweets","Negative Tweets"))
    
#     if tweet_type=='All Tweets':
#         plt.subplots(figsize = (21,7))
#         plt.imshow(st.session_state["wordcloud1"]) # image show
#         plt.axis('off')
#         st.pyplot(plt)
#     if tweet_type=='Positive Tweets':
#         plt.subplots(figsize = (21,7))
#         plt.imshow(st.session_state["wordcloud_ps1"]) # image show
#         plt.axis('off')
#         st.pyplot(plt)
#     # if tweet_type=='Neutral Tweets':
#     #     plt.subplots(figsize = (21,7))
#     #     plt.imshow(st.session_state["wordcloud_nu1"]) # image show
#     #     plt.axis('off')
#     #     st.pyplot(plt)
#     if tweet_type=='Negative Tweets':
#         plt.subplots(figsize = (21,7))
#         plt.imshow(st.session_state["wordcloud_ne1"]) # image show
#         plt.axis('off')
#         st.pyplot(plt)      

#     # st.write("translation complete and time taken is ",(time.time()-start))
# except:
#     pass

if submit:
    with st.spinner('Processing Tweets for topics...'):
        import urllib3, socket
        from urllib3.connection import HTTPConnection

        HTTPConnection.default_socket_options = ( 
            HTTPConnection.default_socket_options + [
            (socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000), #1MB in byte
            (socket.SOL_SOCKET, socket.SO_RCVBUF, 1000000)
        ])
        
        # Initiate UMAP
        umap_model = UMAP(n_neighbors=15, 
                        n_components=5, 
                        min_dist=0.0, 
                        metric='cosine', 
                        random_state=100)
        # Embedding model
        # start = time.time()
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        # st.write("Execution complete and time taken is ",(time.time()-start))
        # Initiate BERTopic
        topic_model = BERTopic(umap_model=umap_model, calculate_probabilities=True)
        # Run BERTopic model
        topics,probs = topic_model.fit_transform(df['cleaned_tweets_tm'])

        # Use the "c-TF-IDF" strategy with a threshold
        new_topics = topic_model.reduce_outliers(df['cleaned_tweets_tm'], topics , strategy="c-tf-idf", threshold=0.1)

        # Reduce all outliers that are left with the "distributions" strategy
        new_topics = topic_model.reduce_outliers(df['cleaned_tweets_tm'], new_topics, strategy="distributions")

        topics_dict = dict(pd.DataFrame(topic_model.get_topic_info())[["Topic","Name"]].values)

        df['topic_final'] = new_topics
        df['topic_name_final'] = df['topic_final'].map(topics_dict)

        st.session_state["tw_data1"] = df

# try:
    # df = st.session_state["tw_data1"]
    # df.to_csv("Latest_Tweet_Summary.csv",index=False)
    # df_sentiment_week_agg = pd.DataFrame(df.groupby(['year_week','topic_name_final'])['number'].count())
    # df_sentiment_week_agg.reset_index(inplace=True)
    # fig3 = px.bar(df_sentiment_week_agg, x="year_week", y="number", color="topic_name_final")
    # fig3.update_layout(title_text="Tweets Topic Modelling",title_font=dict(size=25))
    # st.plotly_chart(fig3, use_container_width=True)


# except:
#     pass

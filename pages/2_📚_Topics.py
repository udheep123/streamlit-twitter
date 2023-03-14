import streamlit as st
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Tweets Topics", page_icon="🌍")

st.markdown("<h1 style='text-align: center; color: black;'>Tweets Topics</h1>", unsafe_allow_html=True)

df = st.session_state["tw_data1"]
df.to_csv("Latest_Tweet_Summary.csv",index=False)
df_sentiment_week_agg = pd.DataFrame(df.groupby(['year_week','topic_name_final'])['number'].count())
df_sentiment_week_agg.reset_index(inplace=True)

# df_sentiment_week_agg_pivot = df_sentiment_week_agg.pivot_table(index=['year_week'], columns='topic_name_final',values='number').fillna(0)
# st.bar_chart(df_sentiment_week_agg_pivot, use_container_width=True)

# st.write(f'<style>.main-svg {{height: 5000px}}</style>', unsafe_allow_html=True)
st.markdown(f'<style>.main-svg {{height: 7000px}}</style>', unsafe_allow_html=True)

fig3 = px.bar(df_sentiment_week_agg, x="year_week", y="number", color="topic_name_final", labels={'number':'# Tweets'})
fig3.update_layout(height=600, xaxis_title=None, legend_title="Tweets Topics")
# fig3.update_layout(title_text="Tweets Topic Modelling",title_font=dict(size=25))
fig3.update_layout(legend=dict(
    orientation="h",y=-0.25))

st.plotly_chart(fig3, use_container_width=True)



# st.write("translation complete and time taken is ",(time.time()-start))

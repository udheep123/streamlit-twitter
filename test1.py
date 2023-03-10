import lang_trans_basic
from multiprocessing import Pool
from googletrans import Translator
import time
import streamlit as st
import pandas as pd

df = pd.read_csv("Latest_Tweet_Summary_Revanth.csv")
df['rawContent'] = [' '.join([ word for word in tweet.split() if not word.startswith('https') ]) for tweet in df['rawContent'].tolist()]    
df['rawContent'] = [' '.join([ word for word in tweet.split() if not word.startswith('http') ]) for tweet in df['rawContent'].tolist()]
df = df[df['rawContent']!=""]
df.reset_index(inplace=True, drop=True)

start = time.time()
st.write("started multiprocessing")

output_tw = []
if (__name__ == '__main__'):
    st.write("working in pool loop")
    pool = Pool()
    for result in pool.imap(lang_trans_basic.lang_chg,df['rawContent'].tolist()):
        output_tw.append(result)
    # pool.close()
    # pool.join()
    
st.write("translation complete and time taken for pool is ",(time.time()-start))

start = time.time()
translator = Translator()
output_tw2 = df['rawContent'].apply(lambda x:translator.translate(x,dest= "en").text)
st.write("translation complete and time taken for apply is ",(time.time()-start))




import ray
import time
start = time.time()
# Start Ray.
ray.init()

@ray.remote
def lang_chg(x):
    from googletrans import Translator
    translator = Translator()
    return translator.translate(x,dest= "en").text

# Start 4 tasks in parallel.
result_ids = []
for i in df['rawContent'].tolist():
    result_ids.append(lang_chg.remote(i))
    
# Wait for the tasks to complete and retrieve the results.
# With at least 4 cores, this will take 1 second.
results = ray.get(result_ids)  # [0, 1, 2, 3]

print(results[10])
print(len(results))
print("translation complete and time taken for apply is ",(time.time()-start))

st.write("Success")
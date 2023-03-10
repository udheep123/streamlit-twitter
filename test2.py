import pandas as pd
import multiprocessing
import time
from lang_trans_basic import lang_chg

start = time.time()
def parallel_apply(df, func,col, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(func, df[col])
    return results

if __name__ == '__main__':
    # Create sample dataframe
    df = pd.read_csv("Latest_Tweet_Summary_Revanth.csv")
    df['rawContent'] = [' '.join([ word for word in tweet.split() if not word.startswith('https') ]) for tweet in df['rawContent'].tolist()]    
    df['rawContent'] = [' '.join([ word for word in tweet.split() if not word.startswith('http') ]) for tweet in df['rawContent'].tolist()]
    df = df[df['rawContent']!=""]
    df.reset_index(inplace=True, drop=True)
    # df = df[['rawContent']]

    # Parallel apply function to each column
    result = parallel_apply(df, lang_chg, 'rawContent')

    # Print result
    # print(result)
    print("translation complete and time taken for apply is ",(time.time()-start))

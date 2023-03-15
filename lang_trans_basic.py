from googletrans import Translator
translator = Translator() 
def lang_chg(x):
    return translator.translate(x,dest= "en").text


import tweetnlp
model_tweetnlp = tweetnlp.load_model('sentiment')
def lang_senti(x):
    return model_tweetnlp.sentiment(str(x))['label']
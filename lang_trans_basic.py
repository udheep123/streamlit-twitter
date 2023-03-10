from googletrans import Translator
translator = Translator() 
def lang_chg(x):
    return translator.translate(x,dest= "en").text
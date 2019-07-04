import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer   
Regexp='[a-zA-Z]+'
tokenizer=RegexpTokenizer(Regexp)
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
wn=WordNetLemmatizer()
stop=set(stopwords.words('english'))
def clean_review(text):
    #text=text.split('<br /><br />')
    text=text.lower()
    text=tokenizer.tokenize(text)
    text=[wn.lemmatize(i) for i in text if i not in stop]
    return ' '.join(text)
    
f=open('./IMDB/imdb_testX.txt','r')
if f.mode == 'r':
    data=f.read()
    data=data.split('<br /><br />')
    #print(len(data))
    val=[clean_review(sent) for sent in data]
    print(len(val))
file1 = open("myfile.txt","w+") 
#L = ["This is Delhi \n","This is Paris \n","This is London \n"]  
  
# \n is placed to indicate EOL (End of Line) 
    #file1.write("Hello \n") 
file1.writelines(val) 
file1.close()

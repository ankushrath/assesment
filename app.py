import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from wordcloud import WordCloud 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import spacy
from collections import Counter

sw=nltk.corpus.stopwords.words('english')

def tokenization(text):
    tokens = re.split(' ',text)
    return tokens

def replace_spaces(x,space,second):
    result = x.replace(space, second)
    return result

sw=nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def prosesing(data):
    copydata=data.copy()
    pattern = r'[' + string.punctuation + ']'
    copydata['text1']=data['text1'].map(lambda m:re.sub(pattern," ",m))
    copydata['text2']=data['text2'].map(lambda m:re.sub(pattern," ",m))
    
    copydata['text1']=copydata['text1'].map(lambda m:m.lower())
    copydata['text2']=copydata['text2'].map(lambda m:m.lower())
    
    copydata['text1']= copydata['text1'].apply(lambda x: tokenization(x))
    copydata['text2']= copydata['text2'].apply(lambda x: tokenization(x))
    
    copydata['text1']=copydata['text1'].apply(lambda x: [item for item in x if item not in sw])
    copydata['text2']=copydata['text2'].apply(lambda x: [item for item in x if item not in sw])
    
    copydata['text1']=copydata['text1'].apply(lambda x: [item for item in x if not item.isdigit()])
    copydata['text2']=copydata['text2'].apply(lambda x: [item for item in x if not item.isdigit()])
    
    copydata['text1']=copydata['text1'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
    copydata['text2']=copydata['text2'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
    
    copydata['text1']=copydata['text1'].apply(lambda x: [item for item in x if item !=''])
    copydata['text2']=copydata['text2'].apply(lambda x: [item for item in x if item !=''])
    
    copydata['text1']=copydata['text1'].apply(lambda x: [item for item in x if len(item) > 1])
    copydata['text2']=copydata['text2'].apply(lambda x: [item for item in x if len(item) > 1])
    
    copydata['text1']= copydata['text1'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
    copydata['text2']= copydata['text2'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
    
    copydata['text1']= copydata['text1'].apply(lambda x: replace_spaces(x,'  ',' '))
    copydata['text2']= copydata['text2'].apply(lambda x: replace_spaces(x,'  ',' '))
    return copydata

def count_vcr(copydata):
    similarity=[]
    for i in range(len(copydata)):
        doc1=copydata['text1'][i]
        doc2=copydata['text2'][i]
        docs=(doc1,doc2)
        matrix = CountVectorizer().fit_transform(docs)
        cosine_sim = cosine_similarity(matrix[0], matrix[1])
        similarity.append(cosine_sim[0][0])
    return similarity




# di = {'text1': 'nuclear body seeks new tech', 'text2': 'terror suspects face arrest'}
# copy_data = prosesing(pd.DataFrame([di]))
# simi = count_vcr(copy_data)


# app.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict_similarity', methods=['POST', 'GET'])
def predict():
    data = request.get_json()
    # text1 = data.get('text1', '')
    # text2 = data.get('text2', '')
    copy_data = prosesing(pd.DataFrame([data]))
    simi = count_vcr(copy_data)
    # Assuming predict_similarity is a function in your_model_module
    # similarity_score = predict_similarity(text1, text2)

    response = {"similarity score": simi}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

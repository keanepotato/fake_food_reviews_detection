# Instead of Random forests, use a different type of decision trees
#%%
#from sklearn.externals import joblib
import joblib
import gradio as gr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings, string
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(text):
    nopunc = [w for w in text if w not in string.punctuation]
    nopunc = ''.join(nopunc)
    return  ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])
def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])

def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

## Showcase the preprocessing algorithm/ code
preprocess_req = False
if preprocess_req:
    df = pd.read_csv('fake reviews dataset.csv')
    stemmer = PorterStemmer()
    
    # Removal of stemming, not necessary
    #df['text_'] = df['text_'].apply(lambda x: stem_words(x))
    lemmatizer = WordNetLemmatizer()

    df["text_"] = df["text_"].apply(lambda text: lemmatize_words(text))

    df['text_'][:10000] = df['text_'][:10000].apply(preprocess)
    df['text_'][10001:20000] = df['text_'][10001:20000].apply(preprocess)
    df['text_'][20001:30000] = df['text_'][20001:30000].apply(preprocess)
    df['text_'][30001:40000] = df['text_'][30001:40000].apply(preprocess)
    df['text_'][40001:40432] = df['text_'][40001:40432].apply(preprocess)
    df['text_'] = df['text_'].str.lower()
    df.to_csv('Preprocessed Fake Reviews Detection Dataset.csv')

### Obtain the preprocessed Dataset

#%%
df = pd.read_csv(r'C:\Users\65932\OneDrive\Desktop\UIT2201\Fake-Reviews-Detection\Preprocessed Fake Reviews Detection Dataset.csv')
df = df.dropna()
review_train, review_test, label_train, label_test = train_test_split(df['text_'],df['label'],test_size=0.35)
print(review_test)


#%%
params = {'classifier__n_estimators':[100,120], 'classifier__min_samples_split': [2,3], 'classifier__min_samples_leaf': [3,5]}
# Other params, 'rf_max_depth': [30,50]
# change min_samples split to float so that you get a fraction
# Find a way to tune the tfidf as well

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfipdf',TfidfTransformer()),
    ('classifier',RandomForestClassifier())
])
print(pipeline.get_params().keys())
randomforest_gridsearch = GridSearchCV(pipeline,param_grid=params, scoring = 'roc_auc', cv=None, verbose = 3) #use randomised search cv instead
randomforest_gridsearch.fit(review_train,label_train)

print(randomforest_gridsearch.best_params_)
print(randomforest_gridsearch.best_score_)
print("Test scores are ",randomforest_gridsearch.score(review_test,label_test))

# pipeline = Pipeline([
#     ('bow',CountVectorizer(analyzer=text_process)),
#     ('tfipdf',TfidfTransformer()),
#     ('classifier',GradientBoostingClassifier)
# ])
#%%
# print(np.where(review_train.isna())[0])

# print(review_train.head())
# print(type(review_train))
# review_train.drop(21567)
# label_train.drop(21567)
# label_train.drop(6659)
# review_train.drop(6659)
# print(np.where(review_train.isna()))
# print(label_train.isnull().values.any())
# print(review_train.isnull().values.any())
# print(review_train)

#%%
print(type(review_test))

#%%
#pipeline.fit(review_train, label_train)
#results = pipeline.predict(review_test)
#print(results)
joblib.dump(randomforest_gridsearch, 'RF_fake_review_detector_gs.pkl')

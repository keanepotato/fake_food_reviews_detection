from cProfile import label
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
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TextClassificationPipeline
import torch
import numpy as np

### IMPORT BERT MODEL

# Load the trained model, must input the model path here
bert_model = AutoModelForSequenceClassification.from_pretrained(r"C:\Users\65932\OneDrive\Desktop\UIT2201\fake_food_reviews_detection\BERTmodel")

# Load the autotokenizer, must input the token path here
bert_tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\65932\OneDrive\Desktop\UIT2201\fake_food_reviews_detection\BERTmodeltoken")

### CREATING BERT PIPELINE
bert_pipeline = TextClassificationPipeline(model=bert_model, tokenizer=bert_tokenizer,return_all_scores=True)

### DEFINE ML FUNCTION
def classify_review(x):
    score_labels = bert_pipeline(x)[0]
    score_labels.sort(key = lambda x : x['score'], reverse=True)
    #print(score_labels)
    label = score_labels[0]['label']
    if label == 'LABEL_1':
        return "Authentic"
    else:
        return "Fake"


### INITIALISE GRADIO INTERFACE
initialise_gradio = True

if initialise_gradio:
    

    txt_1 = gr.Textbox(label= "Review to be examined")

    ### TEST EXAMPLES
    set_one_second_eg = "Restaurant keeps cold drinks cold, hot drinks hot, cold drinks hot."
    
    set_one_first_eg = "Don't try anything 'fusion.' It never tastes good and it's over-priced. I'm told they have excellent Taiwanese dishes.  I don't know what qualifies as an excellent Taiwanese dish, so I won't comment there. The last time I came to Formosa, I had their all-you-can-eat sushi: very disappointing.  Nothing special about their sushi, and I found their menu rather limiting, especially compared to other All-You-Can-Eat sushi places, such as Sushi Palace or Sushi X. Worse, they made us use the same slip of paper over and over again, telling us to just 'cross out' our previous order. This, of course, resulted in confusion for them and some of our orders never came out... or things we previously ordered but did not want a second time around would come. Furthermore, it took forever for the dishes to come out. Don't come for the All-You-Can-Eat sushi or fusion dishes, but if my Taiwanese friends are right, come for their Taiwanese dishes."

    iface = gr.Interface(fn = classify_review, 
                        inputs = txt_1, 
                        outputs = 'label', 
                        title = 'Fake Review Detection with Custom Bert Model',
                        description = 'Input any text that you believe to be fake, this will return an output',
                        article = 
                            '''<div>
                                <p> Hit submit after putting your fake review text to see the results.</p>
                            </div>''',
                        examples = [[set_one_first_eg], [set_one_second_eg]],
                        share = True)

    iface.launch(share = True)
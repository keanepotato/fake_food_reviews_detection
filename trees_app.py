from winreg import REG_RESOURCE_REQUIREMENTS_LIST
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

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])

def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def process_all(text):
    text = preprocess(text)
    text = lemmatize_words(text)
    return text

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

### IMPORT TF-IDF RANDOM FOREST MODEL WITH GRIDSEARCH
print("Running prototype")
loaded_pipeline = joblib.load(r"C:\Users\65932\OneDrive\Desktop\UIT2201\Fake-Reviews-Detection\trained_fake_review_detector.pkl")
# print(loaded_pipeline)
# iter_test = ["Hello World", "hello World", "world hello"]
# print(loaded_pipeline.predict(iter_test)[0])

###DEFINE ML FUNCTION
def classify_review(first, second, third):
    
    iter_text = [process_all(first),process_all(second),process_all(third)]
    print(iter_text)
    results = loaded_pipeline.predict(iter_text)
    result_text = "The first, second and third reviews are: "
    for result in results:
        if result == 0:
            result_text += "fake, "
        else:
            result_text += "authentic, "

    result_text = result_text[:-2]
    result_text += " respectively."           

    return result_text

### INITIALISING THE GRADIO INTERFACE
initialise_gradio = True

if initialise_gradio:

    txt_1 = gr.Textbox(label="First Review")
    txt_2 = gr.Textbox(label="Second Review")
    txt_3 = gr.Textbox(label="Third Review")
   
    ### TEST EXAMPLES
    set_two_first_eg = "Can't miss stop for the best Fish Sandwich in Pittsburgh."
    set_two_second_eg = "Good fish sandwich." 
    set_two_third_eg =  "Let there be no question: Alexions owns the best cheeseburger in the region and they have now for decades. Try a burger on Italian bread. The service is flawlessly friendly, the food is amazing, and the wings? Oh the wings... but it's still about the cheeseburger. The atmosphere is inviting, but you can't eat atmosphere... so go right now. Grab the car keys... you know you're hungry for an amazing cheeseburger, maybe some wings, and a cold beer! Easily, hands down, the best bar and grill in Pittsburgh." 
    
    set_three_first_eg = "What a find! I stopped in here for breakfast while in town for business. The service is so friendly I thought I was down south. The service was quick, frankly and felt like I was with family. \nFantastic poached eggs, Cajun homefries and crispy bacon. Gab and Eat is definitely a place I world recommend to locals. I was stuffed and the bill was only $8.00." 
    set_three_second_eg = "Great little place. Treats you like a local.Eaten here 3 times a week for a month. Same overtime. Barb is always here." 
    set_three_third_eg = "Tonya is super sweet and the front desk people are very helpful" 


    iface = gr.Interface(fn = classify_review, 
                        inputs = [txt_1, txt_2, txt_3], 
                        outputs = 'label', 
                        title = 'Fake Review Detection',
                        description = 'Input any text that you believe to be fake, this will return an output',
                        article = 
                            '''<div>
                                <p> Hit submit after putting your fake review text to see the results.</p>
                            </div>''',
                        examples = [[set_two_first_eg, set_two_second_eg, set_two_third_eg],[set_three_first_eg,set_three_second_eg,set_three_third_eg]],
                        share = True)


    iface.launch(share = True)
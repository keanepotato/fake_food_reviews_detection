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

# Load pipeline containing the model, must input the model path here
loaded_pipeline = joblib.load(r"C:\Users\65932\OneDrive\Desktop\UIT2201\fake_food_reviews_detection\RF_fake_review_detector_gs.pkl")


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
    set_two_third_eg =  "Replacing YOLO Cafe, Toast has an eclectic blend of loose leaf teas & coffee, delicious 'egg toasts' (poached eggs on top of homemade english muffins), sandwiches and other homemade goodies. *Try the egg toast with warm apples, white cheddar cheese, sausage and dijonnaise! AMAZING. I'm so impressed by their revamping of this place, the interior is cozy, clean and perfect for brunch with friends or studying on a Thursday afternoon. The baristas are helpful, friendly and warm. I will be coming here on a weekly basis. Cheers to Toast!" 
    
    set_three_first_eg = "Toast is ok. My goat cheese, tomato and spinach omelet was actually pretty good and the coffee was pretty good too. I had the hummus plate and the hummus was pretty good. But it's kind of expensive and although it's pretty good it's not great. Which is the whole thing. It's fine. There's nothing wrong with it but it's just not exciting in any way. I was kind of pissed that they wouldn't make my kid a plain omelette though when clearly they are capable of doing so. Weird. Nice space." 
    set_three_second_eg = "Great little place. Treats you like a local.Eaten here 3 times a week for a month. Same overtime. Barb is always here." 
    set_three_third_eg = "Tonya is super sweet and the front desk people are very helpful" 


    iface = gr.Interface(fn = classify_review, 
                        inputs = [txt_1, txt_2, txt_3], 
                        outputs = 'label', 
                        title = 'Fake Review Detection via TF-IDF & Random Forests',
                        description = 'Input any text that you believe to be fake, this will return an output',
                        article = 
                            '''<div>
                                <p> Hit submit after putting your fake review text to see the results.</p>
                            </div>''',
                        examples = [[set_two_first_eg, set_two_second_eg, set_two_third_eg],[set_three_first_eg,set_three_second_eg,set_three_third_eg]],
                        share = True)


    iface.launch(share = True)
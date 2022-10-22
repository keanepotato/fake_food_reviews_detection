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

def process_all(text):
    text = clean_text(text)
    text = preprocess(text)
    text = stem_words(text)
    text = lemmatize_words(text)
    return text

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

### Train on fake reviews dataset generically, but deploy on food fake reviews; tf-idf experiments; & then use bert with your own classifier labels

### IMPORT ML MODEL; best use for multiple reviews involving the same reviewer or user
print("Running prototype")
loaded_pipeline = joblib.load(r"C:\Users\65932\OneDrive\Desktop\UIT2201\Fake-Reviews-Detection\trained_fake_review_detector.pkl")
print(loaded_pipeline)
iter_test = ["Hello World", "hello World", "world hello"]
print(loaded_pipeline.predict(iter_test)[0])

initialise_gradio = True

if initialise_gradio:
    def classify_review(first, second, third):
        
        iter_text = [process_all(first),process_all(second),process_all(third)]
        print(iter_text)
        results = loaded_pipeline.predict(iter_text)
        result_text = "The first, second and third reviews are: "
        for result in results:
            if result == "CG":
                result_text += "fake, "
            else:
                result_text += "authentic, "

        result_text = result_text[:-2]
        result_text += " respectively."           

        return result_text
    
    # with gr.Blocks() as demo:
    txt_1 = gr.Textbox(label="First Review")
    txt_2 = gr.Textbox(label="Second Review")
    txt_3 = gr.Textbox(label="Third Review")
    #     txt_output = gr.Textbox(value="", label="Output")
    #     btn = gr.Button(value="Submit")
    #     btn.click(classify_review, inputs=[txt_1, txt_2, txt_3], outputs=[txt_output])
    
    #     gr.Markdown("## Text Examples")
        
    #     ## TEST EXAMPLES
    # set_one_first_eg = "Unfortunately, the frustration of being Dr. Goldberg's patient is a repeat of the experience I've had with so many other doctors in NYC -- good doctor, terrible staff. It seems that his staff simply never answers the phone. It usually takes 2 hours of repeated calling to get an answer. Who has time for that or wants to deal with it? I have run into this problem with many other doctors and I just don't get it. You have office workers, you have patients with medical needs, why isn't anyone answering the phone? It's incomprehensible and not work the aggravation. It's with regret that I feel that I have to give Dr. Goldberg 2 stars." 
    # set_one_second_eg = "Been going to Dr. Goldberg for over 10 years. I think I was one of his 1st patients when he started at MHMG. He's been great over the years and is really all about the big picture. It is because of him, not my now former gyn Dr. Markoff, that I found out I have fibroids. He explores all options with you and is very patient and understanding. He doesn't judge and asks all the right questions. Very thorough and wants to be kept in the loop on every aspect of your medical health and your life." 
    # set_one_third_eg = "Been going to Dr. Goldberg for over 10 years. I think I was one of his 1st patients when he started at MHMG. He's been great over the years and is really all about the big picture. It is because of him, not my now former gyn Dr. Markoff, that I found out I have fibroids. He explores all options with you and is very patient and understanding. He doesn't judge and asks all the right questions. Very thorough and wants to be kept in the loop on every aspect of your medical health and your life."
    
    set_two_first_eg = "Can't miss stop for the best Fish Sandwich in Pittsburgh."
    set_two_second_eg = "Good fish sandwich." 
    set_two_third_eg =  "Let there be no question: Alexions owns the best cheeseburger in the region and they have now for decades. Try a burger on Italian bread. The service is flawlessly friendly, the food is amazing, and the wings? Oh the wings... but it's still about the cheeseburger. The atmosphere is inviting, but you can't eat atmosphere... so go right now. Grab the car keys... you know you're hungry for an amazing cheeseburger, maybe some wings, and a cold beer! Easily, hands down, the best bar and grill in Pittsburgh." 
    
    set_three_first_eg = "What a find! I stopped in here for breakfast while in town for business. The service is so friendly I thought I was down south. The service was quick, frankly and felt like I was with family. \nFantastic poached eggs, Cajun homefries and crispy bacon. Gab and Eat is definitely a place I world recommend to locals. I was stuffed and the bill was only $8.00." 
    set_three_second_eg = "Great little place. Treats you like a local.Eaten here 3 times a week for a month. Same overtime. Barb is always here." 
    set_three_third_eg = "Tonya is super sweet and the front desk people are very helpful" 



    #     gr.Examples([[first_eg], [second_eg],[third_eg]], [txt_1, txt_2, txt_3], txt_output, classify_review, cache_examples=True)


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

    #gr.Markdown("## Text Examples")

    # Adding of examples
    #gr.Examples([["hi", "Adam"], ["hello", "Eve"]], test_fn, cache_examples=True)

    iface.launch(share = True)
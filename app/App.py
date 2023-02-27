import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components
import pickle

with open("app/models/AdaBoost_Classifier.pkl", "rb") as pickle_file:
        ada = pickle.load(pickle_file)
with open("app/models/Decision_Tree_Classifier.pkl", "rb") as pickle_file:
        dtc = pickle.load(pickle_file)
with open("app/models/Logistic_Regression_Classifier.pkl", "rb") as pickle_file:
        lr = pickle.load(pickle_file)
with open("app/models/vectorizer.pkl", "rb") as pickle_file:
        vectorizer = pickle.load(pickle_file)
models = [ada, dtc, lr]

def add_bg_from_url():
    st.markdown(
         f"""
         <DOCTYPE html>
         <html lang='en'>
         <head>
            <meta charset='utf-8'>
            <title>HARMONeY</title>
         </head>
         <body>
            <p id="yKO2mWXkXTnECAKI" style="color:#008037;font-family:YAD-4Fp-fVw-0;line-height:1.4em;text-align:center;"><span id="Vu52fhCqoM983dXA" style="color:#008037;font-size:38px;">HARMONeY</span><br></p>
            <p id="fRhcZc0VZIRi1JHk" style="text-transform:uppercase;color:#008037;font-family:YAD-4Fp-fVw-0;line-height:1.4em;text-align:center;"><span id="aAQSols4Qb3di0LT" style="color:#008037;">BY STREAMLIT AI</span><br></p>
            <style>
               .stApp {{
                  background: rgba(0, 0, 0, 0);
                  background-attachment: fixed;
                  background-size: cover;
                  opacity: 1.0;
               }}
            </style>
         </body>
         </html>
         """,
         unsafe_allow_html=True
     )
   
def tweet_predict(models, tweets):
   outputs = []
   out_tweets = []
   out_prediction = []
   for tweet in tweets:
      for model in models:
         df1 = vectorizer.transform([tweet]).toarray()
         if(model.predict(df1) == 0):
            output = "Hate Speech detected"
         elif(model.predict(df1) == 1):
            output = "Offensive Language detected"
         elif(model.predict(df1) == 2):
            output = "Neither"
         outputs.append(output)
      prediction = pd.Series(outputs).mode()
      out_tweets.append(tweet)
      out_prediction.append(prediction)
      pred = {"Text": out_tweets,
               "Prediction": out_prediction}
   return pred,out_prediction

add_bg_from_url()

tab1, tab2 = st.tabs(['Detector','About'])

font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 16px;
  background-color: white
}
</style>
"""
st.write(font_css, unsafe_allow_html=True)

with tab1:
   st.write("Will that text lead to social reproach? One way to find out:")
   col1, col2 = st.columns(2)
   with col1:
      with st.form('uploaded'):
         st.write('Select .csv file for upload. File should have just one column, which contains text.')
         csv_file = st.file_uploader("Upload text", type = 'csv', accept_multiple_files= False)
         submitted1 = st.form_submit_button("Submit")
   with col2:
      with st.form('inputted'):
         st.write('or just type that text here:')
         txt = st.text_area("Please input your text", height = 148, max_chars= 250)
         submitted2 = st.form_submit_button("Submit")
   
   col3, col4 = st.columns((100,1))
   with col3:
      if submitted1:
         with st.expander("Overview of uploaded .csv file. Showing top 3 rows"):
            st.dataframe(pd.read_csv(csv_file).head(3))
            st.write(pd.read_csv(csv_file).shape)
         df_test = pd.read_csv(csv_file)
         st.write("filename:", csv_file.name)
         st.write(df_test.values.tolist())
         tweets = df_test.values.tolist()
         pred, = tweet_predict(models, tweets)
         st.write(pd.DataFrame(pred))
      elif submitted2: 
         st.write('\"'+ txt + '\":\n\n')
         tweets = []
         tweets.append(txt)
         pred, out_prediction = tweet_predict(models, tweets)
         if str(out_prediction[0].tolist()[0]) == 'Neither':
            st.write('The text has been determined to contain neither offensive language not hate speech.')
         elif str(out_prediction[0].tolist()[0]) == "Offensive Language detected":
            st.write('The text has been determined to contain offensive language.')
         elif str(out_prediction[0].tolist()[0]) == "Hate Speech detected":
            st.write('The text has been determined to contain hate speech.')   
with tab2:
   st.markdown("""
            Offensive language and hate speech has strong linkage to a global increase in physical and psychological violence.
            Social media companies and governments have unfortunately been forced to impose limitations to freedom of speech.
            We aim to provide a tool that will proactively sustain harmony in the social circles by enabling evaluation of text messages.
            Artificial Intelligence techniques have been applied, and you can visit our github repo here for an indepth view of the development.
            """)
   st.subheader("The Visionaries")
   
   # imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")

   # imageUrls = [
   #    "https://images.unsplash.com/photo-1522093007474-d86e9bf7ba6f?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
   #    "https://images.unsplash.com/photo-1610016302534-6f67f1c968d8?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1075&q=80",
   #    "https://images.unsplash.com/photo-1516550893923-42d28e5677af?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=872&q=80",
   #    "https://images.unsplash.com/photo-1541343672885-9be56236302a?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
   #    "https://images.unsplash.com/photo-1512470876302-972faa2aa9a4?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
   #    "https://images.unsplash.com/photo-1528728329032-2972f65dfb3f?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
   #    "https://images.unsplash.com/photo-1557744813-846c28d0d0db?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1118&q=80",
   #    "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
   #    "https://images.unsplash.com/photo-1595867818082-083862f3d630?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
   #    "https://images.unsplash.com/photo-1622214366189-72b19cc61597?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
   #    "https://images.unsplash.com/photo-1558180077-09f158c76707?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
   #    "https://images.unsplash.com/photo-1520106212299-d99c443e4568?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
   #    "https://images.unsplash.com/photo-1534430480872-3498386e7856?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
   #    "https://images.unsplash.com/photo-1571317084911-8899d61cc464?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
   #    "https://images.unsplash.com/photo-1624704765325-fd4868c9702e?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
   #  ]
   # selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)

   # if selectedImageUrl is not None:
   #    st.image(selectedImageUrl)
   col1, col2, col3, col4 = st.columns(4, gap = "small")
   with col1:
      st.markdown('#### Rogers Mugambi')
      st.write("LinkedIn [here](https://www.linkedin.com/in/rogers-mugambi/), \n Github [here](https://github.com/KiomeGM/)")
   with col2:
      st.markdown('#### Vimagh Solomon')
      st.write("LinkedIn [here](http://www.linkedin.com/in/vimagh-solomon-92a84197), \n Github [here](github.com/vimagh/gits)")
   with col3:
      st.markdown('#### Kushieme Kingsman')
      st.write("LinkedIn [here](http://linkedin.com/in/kingsman-kushieme-49155a211), \n Github [here](https://github.com/Kingsmankek)")
   with col4:
      st.markdown('#### Victor Jokanola')
      st.write("LinkedIn [here](https://www.linkedin.com/in/victor-jokanola-87b79a12a), \n Github [here]()")
   col1, col2, col3, col4 = st.columns(4, gap = "small")
   with col1:
      st.markdown('#### Nurudeen Abdulsalaam')
      st.write("LinkedIn [here](http://www.linkedin.com/in/nurudeenabdulsalaam), \n Github [here](https://github.com/KiomeGM/)")
   with col2:
      st.markdown('#### Rhoda Arthur')
      st.write("LinkedIn [here](https://www.linkedin.com/in/rhoda-arthur-6a4348205/), \n Github [here](https://github.com/RhodyArthur)")
   with col3:
      st.markdown('#### Yaswanth Teja Yarlagadda')
      st.write("LinkedIn [here](https://www.linkedin.com/in/yaswanthteja), \n Github [here](https://github.com/yaswanthteja)")
   with col4:
      st.markdown('#### Ketul Patel')
      st.write("LinkedIn [here](https://www.linkedin.com/in/ketul-patel-1a323914b), \n Github [here](https://github.com/ketul6559)")
   col1, col2, col3, col4 = st.columns(4, gap = "small")
   with col1:
      st.markdown('#### Ajayi Daniel')
      st.write("LinkedIn [here](https://www.linkedin.com/in/ajayi-daniel-7b9612252), \n Github [here](https://github.com/spydann)")
   with col2:
      st.markdown('#### Oladimeji Williams')
      st.write("LinkedIn [here](), \n Github [here]()")
   with col3:
      st.markdown('#### M Meenakshi')
      st.write("LinkedIn [here](), \n Github [here]()")
   with col4:
      st.markdown('#### Nahabwe Monica')
      st.write("LinkedIn [here](), \n Github [here]())")		    
      
      
   
   


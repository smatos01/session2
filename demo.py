import pickle
from pathlib import Path
import streamlit_authenticator as stauth  
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud #pip install wordcloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Text Categorisation", page_icon="🧮", layout="wide")

# --- USER AUTHENTICATION ---
names = ["Sandro Matos", "Julia Roberts"]
usernames = ["sandro.matos", "julia.roberts"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

#Create a dictionary for the credentials
credentials = {
"usernames":{
usernames[0]:{"name":names[0],"password":hashed_passwords[0]},
usernames[1]:{"name":names[1],"password":hashed_passwords[1]}            
            }
              }

authenticator = stauth.Authenticate(credentials,"my_app_cookie_demo2", "abcdef_key", cookie_expiry_days=30) 
#Includes a name for cookie to identify the user so they can authenticate without entering their credentials (as well as a random key). 
#number of days you don’t need to login again. If 0 you always need to login.

name, authentication_status, username = authenticator.login("Login", "main") #main or sidebar


if authentication_status == False:
    st.error("Username/password is incorrect") #red box

if authentication_status == None:
    st.warning("Please enter your username and password") #yellow box

if authentication_status: #if true, then run our app

    # ---- SIDEBAR ----
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"So great to have you here, {name}!! 😹 ")


    st.title("🧮️ Text Categorisation")
    st.subheader("Upload a CSV file containing a list of Comments you want to categorise")
    st.markdown ("")

    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
    if uploaded_file:
        st.markdown('---')
        # Read the CSV file into a pandas DataFrame
        df_raw = pd.read_csv(uploaded_file,encoding='cp1252')
        df_raw.Comment = df_raw.Comment.astype("string")
        df_raw.Category = df_raw.Category.astype("string")
        
        st.subheader("Please manually categorise some comments and our Machine Learning model will categorise the rest")
        df = st.experimental_data_editor(df_raw,num_rows="dynamic",use_container_width=True)

        run_model = st.button("Caregorise the remaining comments!",use_container_width=True)

        if run_model:

            # Split the data into training and testing sets
            df_populated = df[~df['Category'].isna()]
            X_train = df_populated['Comment']
            y_train = df_populated['Category']

            df_empty = df[df['Category'].isna()]
            X_test = df_empty['Comment']
            y_test = df_empty['Category']

            # Convert text data into numerical features using TF-IDF vectorization
            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)

            # Train a logistic regression model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Predict the categories for the test data
            y_pred = model.predict(X_test)

            # Check the results
            df_empty['Category'] = y_pred

            st.markdown("")
            st.subheader("Navigate through how our Machine Learning model categorised the comments without a pre-defined category:")
            st.dataframe(df_empty)

            df_all = pd.concat([df_populated, df_empty])
            
            # Create a SentimentIntensityAnalyzer object.
            sid_obj = SentimentIntensityAnalyzer()
            df_all['Sentiment'] = ''

            st.markdown("---")
            
            for cat in df_all['Category'].unique():
                subheader = "Word Cloud for " + cat
                df_all_cat = df_all[df_all.Category == cat]

                
                for i in range(len(df_all_cat['Comment'])):
                    sentiment_dict = sid_obj.polarity_scores(df_all_cat['Comment'].iloc[i])
                    df_all_cat['Sentiment'].iloc[i] = sentiment_dict['compound']
                
                avg_sentiment = round(df_all_cat['Sentiment'].mean(), 2)
                if avg_sentiment > 0.1:
                    emoji = "😀"
                else:
                    if avg_sentiment < -0.1:
                        emoji = "😭"
                    else:
                        emoji = "😐"

                text = ' '.join(df_all_cat['Comment'])
                wordcloud = WordCloud().generate(text)
                
                st.subheader(subheader)
                st.markdown(f"Sentiment Score: {avg_sentiment} {emoji}")
                
                # Display the generated image
                fig, ax = plt.subplots(figsize = (12, 8))
                ax.imshow(wordcloud)
                plt.axis("off")
                st.pyplot(fig)
                



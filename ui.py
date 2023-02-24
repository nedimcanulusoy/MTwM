import datetime
import random
import json

import pandas as pd
import streamlit as st

from data_processing import return_file, return_df
from model import model_training, predict

# Define global variables
POSITIVE = "Positive"
NEGATIVE = "Negative"


def user_input():
    # Get user input, do encoding stuffs and then predict
    st.set_page_config(page_title="Music and Mental Health", page_icon=":musical_note:", layout="wide")

    st.title("Music and Mental Health 🎶 | 💭")
    st.write('Curious about the effect of music on your mental health? You can find the answer here!')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=90)
        primary_streaming_service = st.selectbox("Which Streaming Service do you use?",
                                                 ["Spotify", "YouTube Music", "Apple Music",
                                                  "I do not use a streaming service.",
                                                  "Pandora",
                                                  "Other streaming service"])

        hours_per_day = st.number_input("Hours per day spent listening to music", min_value=0, max_value=24)
        while_working = st.selectbox("Do you listen to music while working?", ["Yes", "No"])
        instrumentalist = st.selectbox("Are you an instrumentalist?", ["Yes", "No"])
        composer = st.selectbox("Are you a composer?", ["Yes", "No"])
        fav_genre = st.selectbox("Favourite Genre",
                                 ["Classical", "Country", "EDM", "Folk", "Gospel", "Hip-Hop", "Jazz", "K-Pop", "Latin",
                                  "Lofi", "Metal", "Pop", "R&B", "Rap", "Rock", "Video Game Music"])
        exploratory = st.selectbox("Do you enjoy exploring new music?", ["Yes", "No"])
    with col2:
        foreign_languages = st.selectbox("Do you listen to music in any foreign languages?", ["Yes", "No"])
        bpm = st.number_input("Slow & Fast Scale", min_value=0, max_value=300)
        frequency_classical = st.selectbox("How often do you listen to Classical music?",
                                           ["Very frequently", "Sometimes", "Rarely", "Never"], key=1)
        frequency_country = st.selectbox("How often do you listen to Country music?",
                                         ["Very frequently", "Sometimes", "Rarely", "Never"], key=2)
        frequency_edm = st.selectbox("How often do you listen to EDM music?",
                                     ["Very frequently", "Sometimes", "Rarely", "Never"], key=3)
        frequency_folk = st.selectbox("How often do you listen to Folk music?",
                                      ["Very frequently", "Sometimes", "Rarely", "Never"], key=4)
        frequency_gospel = st.selectbox("How often do you listen to Gospel music?",
                                        ["Very frequently", "Sometimes", "Rarely", "Never"], key=5)
        frequency_hip_hop = st.selectbox("How often do you listen to Hip Hop music?",
                                         ["Very frequently", "Sometimes", "Rarely", "Never"], key=6)
    with col3:
        frequency_jazz = st.selectbox("How often do you listen to Jazz music?",
                                      ["Very frequently", "Sometimes", "Rarely", "Never"], key=7)
        frequency_k_pop = st.selectbox("How often do you listen to K-Pop music?",
                                       ["Very frequently", "Sometimes", "Rarely", "Never"], key=8)
        frequency_latin = st.selectbox("How often do you listen to Latin music?",
                                       ["Very frequently", "Sometimes", "Rarely", "Never"], key=9)
        frequency_lofi = st.selectbox("How often do you listen to Lo-Fi music?",
                                      ["Very frequently", "Sometimes", "Rarely", "Never"], key=10)
        frequency_metal = st.selectbox("How often do you listen to Metal music?",
                                       ["Very frequently", "Sometimes", "Rarely", "Never"], key=11)
        frequency_pop = st.selectbox("How often do you listen to Pop music?",
                                     ["Very frequently", "Sometimes", "Rarely", "Never"], key=12)
        frequency_rnb = st.selectbox("How often do you listen to R&B music?",
                                     ["Very frequently", "Sometimes", "Rarely", "Never"], key=13)
        frequency_rap = st.selectbox("How often do you listen to Rap music?",
                                     ["Very frequently", "Sometimes", "Rarely", "Never"], key=14)
    with col4:
        frequency_rock = st.selectbox("How often do you listen to Rock music?",
                                      ["Very frequently", "Sometimes", "Rarely", "Never"], key=15)
        frequency_video_game_music = st.selectbox("How often do you listen to Video Games music?",
                                                  ["Very frequently", "Sometimes", "Rarely", "Never"], key=16)
        anxiety = st.number_input("Anxiety (0-10)", min_value=0, max_value=10)
        depression = st.number_input("Depression (0-10)", min_value=0, max_value=10)
        insomnia = st.number_input("Insomnia (0-10)", min_value=0, max_value=10)
        ocd = st.number_input("OCD (0-10)", min_value=0, max_value=10)

        music_effects = st.selectbox("Have you ever used music as a coping mechanism for difficult emotions?",
                                     ["Yes", "No"])
        permissions = st.selectbox("Do you have permission to use this data under GDPR rules?", ["Yes", "No"])

    current_time = datetime.datetime.now()

    input_data = {
        'Timestamp': current_time,
        'Age': age,
        'Primary streaming service': primary_streaming_service,
        'Hours per day': hours_per_day,
        'While working': while_working,
        'Instrumentalist': instrumentalist,
        'Composer': composer,
        'Fav genre': fav_genre,
        'Exploratory': exploratory,
        'Foreign languages': foreign_languages,
        'BPM': 0,
        'Frequency [Classical]': frequency_classical,
        'Frequency [Country]': frequency_country,
        'Frequency [EDM]': frequency_edm,
        'Frequency [Folk]': frequency_folk,
        'Frequency [Gospel]': frequency_gospel,
        'Frequency [Hip hop]': frequency_hip_hop,
        'Frequency [Jazz]': frequency_jazz,
        'Frequency [K pop]': frequency_k_pop,
        'Frequency [Latin]': frequency_latin,
        'Frequency [Lofi]': frequency_lofi,
        'Frequency [Metal]': frequency_metal,
        'Frequency [Pop]': frequency_pop,
        'Frequency [R&B]': frequency_rnb,
        'Frequency [Rap]': frequency_rap,
        'Frequency [Rock]': frequency_rock,
        'Frequency [Video game music]': frequency_video_game_music,
        'Anxiety': anxiety,
        'Depression': depression,
        'Insomnia': insomnia,
        'OCD': ocd,
        'Music effects': music_effects,
        'Permissions': 'I understand.'
    }
    user_data = pd.DataFrame(input_data, index=[0])
    return user_data


user_data = user_input()

file_ = return_file()
# Add this to the end of the main dataframe
file = file_.append(user_data, ignore_index=True)

df = return_df()
mtwm_model = model_training()


def app():
    if st.button("Predict my situtation!"):
        new_data = df.tail(1).drop(["MUSIC_EFFECTS"], axis=1)
        prediction = predict(mtwm_model, new_data)
        if prediction == 1:
            st.info(f"Your predicted music therapy situation is: {POSITIVE}")
            st.info("You are in a situation where you could benefit from music therapy!")

            # Make music suggestions based on user favourite genre
            st.info("Here are some songs that you might like:")
            with open("music.json", "r") as f:
                music = json.load(f)

            favorite_genre = file.tail(1)['FAV_GENRE'].values[0]
            if favorite_genre in music:
                for i, recommendation in enumerate(music[favorite_genre][:3], start=1):
                    st.warning(f"{i}. {recommendation}")

        else:
            st.error(f"Your predicted music therapy situation is: {NEGATIVE}")
            st.error(
                "We can't guarantee that musiotherapy will help you in your current situation, therefore we recommend that you go to a psychologist for more effective support.")
            with open("dummy_psychologists.json") as f:
                psychologists = json.load(f)

            psychologist = random.choice(list(psychologists.values()))
            st.warning(f"We recommend you to contact {psychologist['name']} for more effective support.")
            st.warning(f"Here is his/her contact information: {psychologist['contact']}")
            st.warning(f"Here is his/her website: {psychologist['website']}")
            st.warning(f"Here is his/her address: {psychologist['address']}")
            st.warning(f"Here is his/her country: {psychologist['country']}")


st.markdown(
    """
<style>
.main {
background-color: #395675;
font-size: 14px;
}
.reportview-container .main .block-container{
    max-width: 1000px;
    padding-top: 10px;
    padding-right: 10px;
    padding-left: 10px;
    padding-bottom: 10px;
}


.stButton button {
  color: white; /* change the text color */
  padding: 10px 20px; /* adjust the padding */
  border-radius: 8px; /* adjust the border radius */
  font-size: 20px; /* adjust the font size */
  cursor: pointer; /* change the cursor on hover */ 
}

.stButton button:hover {
  background-color: #1b3a4b; /* change the background color on hover */
}

p {
  font-size: 16px;
  font-family: Arial, sans-serif;
}

a {
  color: blue;
  text-decoration: none;
}

a:hover {
  color: darkblue;
  text-decoration: underline;
}


</style>
""",
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    app()

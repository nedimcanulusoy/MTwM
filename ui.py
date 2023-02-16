import datetime
import random

import pandas as pd
import streamlit as st

from data_processing import return_file, return_df
from model import model_training, predict


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
    if st.button("Predict My Situation"):
        # Get the last row of the dataframe as a new dataframe and make prediction with it and drop MUSIC_EFFECTS column
        new_data = df.tail(1).drop(["MUSIC_EFFECTS"], axis=1)
        prediction = predict(mtwm_model, new_data)
        if prediction == 1:
            st.info("Music can help you improve your mental health, which is why we recommend getting musiotherapy.")
        else:
            st.info(
                "We can't guarantee that musiotherapy will help you in your current situation, therefore we recommend that you go to a psychologist for more effective support.")

        values = ['Latin', 'Rock', 'Video game music', 'Jazz', 'R&B', 'K pop', 'Country', 'EDM', 'Hip hop', 'Pop',
                  'Rap', 'Classical', 'Metal', 'Folk', 'Lofi', 'Gospel']

        music_dict = {value: [] for value in values}

        # Assume the most popular music in each category is as follows
        music_dict['Latin'] = ['Despacito', 'Bailando', 'La Bicicleta']
        music_dict['Rock'] = ['Stairway to Heaven', 'Bohemian Rhapsody', 'Hotel California']
        music_dict['Video game music'] = ['Final Fantasy VII Main Theme', 'Super Mario Bros. Theme', 'Halo Theme']
        music_dict['Jazz'] = ['Take the A Train', 'Misty', 'Round Midnight']
        music_dict['R&B'] = ['Sexual Healing', 'I Want You Back', 'Billie Jean']
        music_dict['K pop'] = ['Gangnam Style', 'Dynamite', 'Butter']
        music_dict['Country'] = ['Friends in Low Places', 'Amarillo by Morning', 'I Walk the Line']
        music_dict['EDM'] = ['Levels', 'Silent Shout', 'Strobe']
        music_dict['Hip hop'] = ['Rapper\'s Delight', 'N.Y. State of Mind', 'Started From the Bottom']
        music_dict['Pop'] = ['Billie Jean', 'I Want It That Way', 'Naked by James Arthur']
        music_dict['Rap'] = ['Rapper\'s Delight', 'N.Y. State of Mind', 'M.v.k Sıkıntı Misafirim']
        music_dict['Classical'] = ['Beethoven\'s Symphony No. 5', 'Mozart\'s Symphony No. 40',
                                   'Bach\'s Brandenburg Concerto No. 3']
        music_dict['Metal'] = ['Black Sabbath', 'Master of Puppets', 'Hallowed Be Thy Name']
        music_dict['Folk'] = ['This Land Is Your Land', 'Blowin\' in the Wind', 'The Times They Are A-Changin\'']
        music_dict['Lofi'] = ['Lofi Girl', 'Rainy Jazz', 'Chillhop Cafe']
        music_dict['Gospel'] = ['Amazing Grace', 'Oh Happy Day', 'Total Praise']

        psychologists = [
            {
                "name": "Dr. Adrian Johansson",
                "country": "Sweden",
                "contact": "+46 123 456 7890",
                "address": "Vasagatan 12, Stockholm",
                "website": "www.adrianpsychologist.se"
            },
            {
                "name": "Dr. Emre Caldemir",
                "country": "Turkey",
                "contact": "+90 212 345 6789",
                "address": "Istiklal Caddesi 123, Istanbul",
                "website": "www.emrepsychologist.com"
            },
            {
                "name": "Dr. Emma Andersson",
                "country": "Sweden",
                "contact": "+46 123 456 7890",
                "address": "Sveavägen 34, Gothenburg",
                "website": "www.emmapsychologist.se"
            },
            {
                "name": "Dr. Ahsen Kaya",
                "country": "Turkey",
                "contact": "+90 212 345 6789",
                "address": "Taksim Square 456, Istanbul",
                "website": "www.ahsenpsychologist.com"
            }
        ]

        random_psychologist = random.choice(psychologists)

        if prediction == 1:
            for key, value in music_dict.items():
                if key == file.tail(1)['FAV_GENRE'].values[0]:
                    st.info(
                        f"Here are our recommendations for {key} category music that will instantly make you better")
                    for i in range(3):
                        st.info(f"{i + 1}. {value[i]}")
        elif prediction == 0:
            for key, value in music_dict.items():
                if key == file.tail(1)['FAV_GENRE'].values[0]:
                    st.info(
                        f"However, our music recommendations on {key} category that will make feel good to you for a short time period!")
                    st.info(f"Here is a psychologist that we recommend for you: {random_psychologist['name']}")
                    st.info(f"Country: {random_psychologist['country']}")
                    st.info(f"Contact: {random_psychologist['contact']}")
                    st.info(f"Address: {random_psychologist['address']}")
                    st.info(f"Website: {random_psychologist['website']}")
                    for i in range(3):
                        st.write(f"{i + 1}. {value[i]}")


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

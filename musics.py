import json

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

# Convert the dictionary into a JSON string
json_string = json.dumps(music_dict, indent=4)

# Save the JSON string into a file
with open('music.json', 'w') as json_file:
    json_file.write(json_string)

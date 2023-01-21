#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.1f' % x)


# In[19]:


#Load the data
df = pd.read_csv("mxmh_survey_results.csv")
#View the data
df.head(10)


# In[20]:


# Obsessive-compulsive disorder (OCD) 
# Beats per minute of favorite genre (BPM) : Favori türün dakikadaki vuruş sayısı


# In[21]:


df.columns = [col.replace(" ","_") for col in df.columns] 


# In[22]:


df.replace(['No', 'Yes'],[0, 1], inplace=True)
df.head()


# In[23]:


def quick_info(dataframe):
    print("--------- SHAPE ---------")
    print(dataframe.shape)
    print("--------- COLUMNS ---------")
    print(dataframe.columns)
    print("--------- INFO ---------")
    print(dataframe.info())
    print("--------- FREQUENCY ---------")
    print(dataframe.nunique())
    print("--------- ANY NULL VALUES ---------")
    print(dataframe.isnull().values.any())
    print("--------- SUM OF NULL VALUES ---------")
    print(dataframe.isnull().sum())
    print("--------- DESCRIBE ---------")
    print(dataframe.describe().T)

quick_info(df)


# In[24]:


df['MH_SCORE'] = df['Anxiety'] + df['Depression'] + df['Insomnia'] + df['OCD']


# In[25]:


df.describe(exclude='number').T


# In[26]:


df['Fav_genre'].value_counts()


# In[27]:


#unique values
df['Fav_genre'].unique()


# In[28]:


df['Primary_streaming_service'].unique()


# In[30]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=df, x="MH_SCORE", y="Age", alpha = 0.9, marker = "X", color = "sienna")
plt.title('Age Distribution by Mental Health Score')
plt.xlabel('Mental Health')
plt.ylabel('Age')
plt.xticks(rotation=45, ha="right")
plt.show()


# In[32]:


df['Age'].plot.hist(bins=30, grid=True, color='darkred')
plt.title('Age')
plt.show()


# Katilimcilarin genel olarak 15-30 yas arasi oldugu gorunuyor

# In[33]:


# specify the bin edges
bin_edges = [df["Age"].min(), 19, 30, 55, df["Age"].max()]

# create labels
labels = ['Child', 'Young', 'Adult', 'Elderly']
df["AGE_BINS"] = df["AGE_BINS"] = pd.cut(df["Age"], bins=bin_edges, labels=labels)


# In[34]:


df.head()


# In[35]:


age_bins_counts = df['AGE_BINS'].value_counts(normalize=True)
age_bins_counts.plot.pie(autopct='%1.1f%%', labels=age_bins_counts.index)
plt.show()


# 10-18 => Child 19 - 24 => Young 25 - 44 => Adult 45 - 89 => Elderly

# In[36]:


df.groupby("AGE_BINS").agg({"Age":["min","max",'mean']})


# In[37]:


plt.figure(figsize=(6,4))
sns.countplot(x=df['Primary_streaming_service'])
plt.xticks(rotation=75)


# In[40]:


s_colors = ['lightgreen', 'lightcoral', 'steelblue', 'palevioletred', 'gold', 'darkturquoise']

services = df['Primary_streaming_service'].value_counts(normalize=True)
services.plot(kind='pie', colors = s_colors, autopct='%1.1f%%')

plt.title('Streaming services by popularity')
plt.ylabel("")


# In[44]:


df.replace(['Other streaming service', 'I do not use a streaming service.', 'YouTube Music'],
                       ['Other', 'None', 'YouTube'], inplace=True)

bplot = sns.boxplot(data=df, x="Primary_streaming_service", y = "Age",
            showfliers = False)

plt.title('Streaming services by Age')


# In[46]:


age_services = df.groupby('Primary_streaming_service').agg({"Age":["min","max", 'mean']})
age_services.plot.bar(edgecolor="black")
age_services


# In[47]:


fig = plt.figure(figsize=(6,2))

plt.suptitle("Preference")

ax = fig.add_subplot(121)

inst = df['Exploratory'].value_counts()
inst.plot(kind='pie', colors = ["blue", "pink"], labeldistance = 1.2)

ax = fig.add_subplot(122)

comp = df['Foreign_languages'].value_counts()
comp.plot(kind='pie', colors = ["blue", "pink"], labeldistance = 1.2)

fig = plt.figure(figsize=(6,2))

plt.suptitle("WHILE WORKING/STUDYING & FOREIGN LANGUAGES")

ax = fig.add_subplot(121)

inst = df['Foreign_languages'].value_counts()
inst.plot(kind='pie', colors = ["blue", "pink"], labeldistance = 1.2)

ax = fig.add_subplot(122)

comp = df['While_working'].value_counts()
comp.plot(kind='pie', colors = ["blue", "pink"], labeldistance = 1.2)


# Yeni muzik kesfedenlerin orani yuksek iken kesiflerini yabanci dilde yapma oranlari neredeyse yari yariya.
# 
# Bir isle mesgul olurken muzik dinleyenlerin orani cok yuksek ancak bunu yabanci dilden yana tercih etme oranlari yari yariya

# In[49]:


df.hist(['Hours_per_day'], bins='auto', density=True, color = 'slateblue', grid=False, edgecolor='black')
plt.ylabel('Listeners Density')
plt.xlabel('Hours')
plt.show()


# In[50]:


df['Hours_per_day'].value_counts().loc[lambda x : x>100]


# In[53]:


fig = plt.figure(figsize=(6,2))

plt.suptitle("Musical background")

ax = fig.add_subplot(121)

inst = df['Instrumentalist'].value_counts()
inst.plot(kind='pie', colors = ["orange", "indianred"], labeldistance = 1.2)

ax = fig.add_subplot(122)

comp = df['Composer'].value_counts()
comp.plot(kind='pie', colors = ["orange", "indianred"], labeldistance = 1.2)


# In[55]:


labels = ['Anxiety', 'Depression','Insomnia', 'OCD']
x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots(figsize=(8,4))

b1 = ax.bar(x-2*width, df[(df.Instrumentalist == 0)].median()[-4:], width, label = "Non Instrumentalist")
b2 = ax.bar(x-width, df[(df.Instrumentalist == 1)].median()[-4:], width, label = "Instrumentalist")
b3 = ax.bar(x, df[(df.Composer == 0)].median()[-4:], width, label = "Non Composer")
b4 = ax.bar(x+width, df[(df.Composer == 1)].median()[-4:], width, label = "Composer")

ax.set_ylim([0, 8])
ax.set_ylabel('Ranking')
ax.set_title('Mental health ranking distribution')
ax.set_xticks(x, labels)
ax.legend()

plt.show()


# In[56]:


corr_mt = df.corr()
corr_matrix = corr_mt.iloc[:-5 , :-5]

mask = np.triu(corr_matrix)

sns.heatmap(corr_matrix, annot=True, mask=mask, cmap = sns.cm.rocket_r)
plt.show()


# In[59]:


plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Classical]'])
plt.xlabel('Classical Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Country]'])
plt.xlabel('Country Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[EDM]'])
plt.xlabel('EDM Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Folk]'])
plt.xlabel('Folk Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Gospel]'])
plt.xlabel('Gospel Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Hip_hop]'])
plt.xlabel('Hip Hop Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Jazz]'])
plt.xlabel('Jazz Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[K_pop]'])
plt.xlabel('K Pop Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Latin]'])
plt.xlabel('Latin Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Lofi]'])
plt.xlabel('Lofi Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Metal]'])
plt.xlabel('Metal Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Pop]'])
plt.xlabel('Pop Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[R&B]'])
plt.xlabel('R&B Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Rap]'])
plt.xlabel('Rap Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Rock]'])
plt.xlabel('Rock Music')

plt.figure(figsize=(5,2))
sns.countplot(x=df['Frequency_[Video_game_music]'])
plt.xlabel('Video Game Music')


# In[60]:


plt.figure(figsize=(6,4))
sns.lineplot(x=df['Fav_genre'], y=df['Age'], ci=None)
plt.xticks(rotation=90)


# In[61]:


sns.scatterplot(data=df, y="Fav_genre", x="Age", alpha = 0.5, marker = "X", color = "sienna")
plt.title('Age distribution by genre');


#  Rock, en çeşitli yaş aralığına sahiptir. 
#  Klasik ve Pop dinleyicileri de diğer türlere göre daha geniş bir yaş aralığına sahiptir. 
#  K pop ve Lofi gibi bazı müzik türlerinin daha spesifik ve daha genç bir yaş grubuna hitap ettiği görülüyor.

# In[62]:


plt.figure(figsize=(6,4))
sns.boxplot(x=df['Fav_genre'], y=df['Hours_per_day'])
plt.xticks(rotation=90)


# In[63]:


plt.figure(figsize=(6,4))
sns.barplot(x=df['Fav_genre'], y=df['Anxiety'], errwidth=0)
plt.xticks(rotation=90)


# In[64]:


plt.figure(figsize=(6,4))
sns.barplot(x=df['Fav_genre'], y=df['Insomnia'], errwidth=0)
plt.xticks(rotation=90)


# In[65]:


plt.figure(figsize=(6,4))
sns.barplot(x=df['Fav_genre'], y=df['OCD'], errwidth=0)
plt.xticks(rotation=90)


# In[66]:


plt.figure(figsize=(6,4))
sns.barplot(x=df['Fav_genre'], y=df['Depression'], errwidth=0)
plt.xticks(rotation=90)


# In[67]:


plt.figure(figsize=(6,4))
sns.barplot(x=df['Fav_genre'], y=df['Age'], hue=df['Music_effects'], errwidth=0, palette='coolwarm')
plt.xticks(rotation=90)


# In[68]:


plt.figure(figsize=(6,4))
sns.barplot(x=df['Fav_genre'], y=df['OCD'], hue=df['Music_effects'], errwidth=0, palette='coolwarm')
plt.xticks(rotation=90)


# In[69]:


plt.figure(figsize=(6,4))
sns.barplot(x=df['Fav_genre'], y=df['Anxiety'], hue=df['Music_effects'], errwidth=0, palette='coolwarm')
plt.xticks(rotation=90)


# In[70]:


plt.figure(figsize=(6,4))
sns.barplot(x=df['Fav_genre'], y=df['Insomnia'], hue=df['Music_effects'], errwidth=0, palette='coolwarm')
plt.xticks(rotation=90)


# In[71]:


plt.figure(figsize=(6,4))
sns.barplot(x=df['Fav_genre'], y=df['Depression'], hue=df['Music_effects'], errwidth=0, palette='coolwarm')
plt.xticks(rotation=90)


# In[72]:


mental_all = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

mental_df = df[mental_all]
mental_df.round(0).astype(int)

disorder_count = []
for disorder in mental_all:
    x=0
    while x !=11:
        count =  (mental_df[disorder].values == x).sum()
        disorder_count.append(count)
        x +=1

labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots(figsize=(13, 9))

b1 = ax.bar(x-2*width, disorder_count[0:11], width, label="Anxiety", color = 'lightpink')
b2 = ax.bar(x-width, disorder_count[11:22], width, label="Depression", color = 'cornflowerblue')
b3 = ax.bar(x, disorder_count[22:33], width, label="Insomnia", color = 'darkmagenta')
b4 = ax.bar(x+width, disorder_count[33:], width, label="OCD", color = 'orange')

ax.set_ylim([0, 170])
ax.set_ylabel('Number of Rankings')
ax.set_xlabel('Ranking')
ax.set_title('Mental health ranking distribution')
ax.set_xticks(x, labels)
ax.legend()

plt.show()


# In[73]:


for disorder in mental_all:
    avg_disorder = str(round(df[disorder].mean(), 2))
    print(f"{disorder} AVERAGE: {avg_disorder}")


# Depresyon ve anksiyete için yüksek sıralamalar çok yaygındır ve sırasıyla ortalama 5 ve 6'dır. Her sıralama arasında popülerlik de aynı yönde hareket ediyor gibi görünüyor (yani, 1. sıradan 2. sıraya kadar, bu sıralamaların popülaritesi hem anksiyete hem de depresyon için artıyor.) İlginçtir ki (endişe verici olsa da), belirli bir bireyin depresyonu sıralaması daha muhtemeldir. 0'dan 10'da.
# 
# Uykusuzluk biraz yaygındır ve 0 sıralamasının dışında daha adil bir şekilde dağıtılır. Bununla birlikte, uykusuzluk sıralamaları, sıralamalar yükseldikçe popülaritesinde düşüş eğilimi göstermektedir.
# 
# Mod olarak 0 ile OKB en az görülen bozukluktur. Uykusuzluk sıralamasına benzer şekilde, OKB sıralaması yükseldikçe popülaritesini düşürme eğilimindedir.

# In[74]:


def extreme_LT_MH(anx_th=0, depr_th=0, ins_th=0, ocd_th=0):
    mental_all = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

    anxiety_extreme = df.loc[(df['Anxiety'] > anx_th)].mean()[1]
    depression_extreme = df.loc[(df['Depression'] > depr_th)].mean()[1]
    insomnia_extreme = df.loc[(df['Insomnia'] > ins_th)].mean()[1]
    ocd_extreme = df.loc[(df['OCD'] > ocd_th)].mean()[1]

    extreme_means = [anxiety_extreme, depression_extreme, insomnia_extreme, ocd_extreme]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(mental_all, extreme_means, color=('lightpink', 'cornflowerblue', 'darkmagenta', 'orange'))

    ax.set_xlabel("Average hours listened", fontsize=12)
    ax.set_title("Hours listened for individuals with extreme MH rankings", fontsize=14)
    ax.set_xlim(3,5.5)
    ax.grid(axis='x', color='grey', linestyle='-.', linewidth=0.5)
    for i, v in enumerate(extreme_means):
        ax.text(v, i, str(round(v,2)), color='black', fontweight='bold', fontsize=12, ha='center')
    plt.show()
    
extreme_LT_MH(8,8,8,8)


def low_LT_MH(anx_th=0, depr_th=0, ins_th=0, ocd_th=0):
    mental_all = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

    anxiety_low = df.loc[(df['Anxiety'] < anx_th)].mean()[1]
    depression_low = df.loc[(df['Depression'] < depr_th)].mean()[1]
    insomnia_low = df.loc[(df['Insomnia'] < ins_th)].mean()[1]
    ocd_low = df.loc[(df['OCD'] < ocd_th)].mean()[1]

    low_means = [anxiety_low, depression_low, insomnia_low, ocd_low]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(mental_all, low_means, color=('lightpink', 'cornflowerblue', 'darkmagenta', 'orange'))

    ax.set_xlabel("Average hours listened", fontsize=12)
    ax.set_title("Hours listened for individuals with low MH rankings", fontsize=14)
    ax.set_xlim(3,4)
    ax.grid(axis='x', color='grey', linestyle='-.', linewidth=0.5)
    for i, (j, v) in enumerate(zip(mental_all, low_means)):
        ax.text(v, i, str(round(v,2)), color='black', fontweight='bold', fontsize=12, ha='center')
    plt.show()

low_LT_MH(1,1,1,1)


# In[75]:


corr_matrix = corr_mt.iloc[8: , 8:]
mask = np.triu(corr_matrix)

sns.heatmap(corr_matrix, annot=True, mask=mask, cmap = sns.cm.rocket_r)
plt.show()


# In[77]:


plt.figure(figsize=(5,4))
plt.title('Effects of Music on Mental Health')

effects = df['Music_effects'].value_counts()
effects.plot(kind='pie', colors = ["purple", "lime", "orange"], ylabel= '');


# In[78]:


effects


#  Ankete katılanların çoğu, müziğin ruh sağlıkları üzerinde olumlu bir etkisi olduğunu düşünüyor. 
#  Geri kalan katılımcılardan sadece 15 kişi müziğin olumsuz bir etkisi olduğunu bildirdi.

# In[80]:


effects_on_personas = df.groupby("Music_effects").agg({"AGE_BINS":"value_counts"})
effects_on_personas


# Mental sagliginda iyilesme gorenlerin buyuk cogunlugu 18 yas alti cocuklar, ardindan da gencler.
# 
# Mental sagliginda birsey degismeyenler ise genellikle cocuk, genc ve yetiskinler.

# In[81]:


effects_on_personas.unstack().plot(kind='bar', stacked=True)
plt.xlabel("Music Effects")
plt.ylabel("Frequency")
plt.show()


# In[82]:


import matplotlib.pyplot as plt
import seaborn as sns

genre = df["Fav_genre"].value_counts().loc[lambda x: x>10]

plt.figure(figsize=(8,8))

plt.pie(genre, labels=genre.index, labeldistance = 1.2, 
        explode=[0.05 if i<13 else 0 for i in range(len(genre))], 
        colors = sns.color_palette('pastel')[0:len(genre)],
        autopct='%1.1f%%', pctdistance=0.8, startangle=0)

plt.title('Top genre breakdown', fontsize=20)
plt.legend(fontsize=12, bbox_to_anchor=(1,0.8))
plt.axis('equal')
plt.show()


# In[85]:


g_all = df['Fav_genre'].unique()
g_all.sort()
fg_df = df.groupby(['Fav_genre'])
fg_dist = fg_df['Music_effects'].value_counts(ascending=False, normalize=True).tolist()

insert_indices = [5, 8, 11, 13, 14, 17, 20, 23, 26, 28, 29, 32, 38]
for i in range(len(insert_indices)):
    fg_dist.insert(insert_indices[i], 0)

imp_dist = fg_dist[0::3]
no_eff_dist = fg_dist[1::3]
wors_dist = fg_dist[2::3]

width = 0.22

x = np.arange(len(g_all))

fig, ax = plt.subplots(figsize=(13, 9))

b1 = ax.bar(x-width, imp_dist, width, label="Improve", color = 'indianred')
b2 = ax.bar(x, no_eff_dist, width, label="No effect", color = 'gold')
b3 = ax.bar(x+width, wors_dist, width, label="Worsen", color = 'darkblue')

plt.title("Music effects by Favorite Genre")
ax.set_ylabel('Distribution')
ax.set_xlabel('Genre')
ax.set_xticks(x, g_all, rotation = 45)
ax.legend()

plt.show()


# In[93]:


print('Favorite genres of highest combined MH scorers:')
df.nlargest(60, ['MH_SCORE'])['Fav_genre'].value_counts()


# In[94]:


print('Favorite genres of lowest combined MH scorers:')
df.nsmallest(60, ['MH_SCORE'])['Fav_genre'].value_counts()


# In[95]:


df.nlargest(1, ['MH_SCORE'])['Fav_genre']


# In[ ]:





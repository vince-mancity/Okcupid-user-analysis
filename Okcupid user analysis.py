from numpy.lib.shape_base import _kron_dispatcher
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from mlxtend.preprocessing import TransactionEncoder
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
import matplotlib.ticker as ticker
from langdetect import detect, detect_langs
import gower
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import spacy
from spacy.lang.en import English


cupid = pd.read_csv("C:/Users/Phoenix/Desktop/okcupid.csv")
df = cupid.copy()
df.info()

### ===================================age=================================== ###
df.age.isna().sum()
df.age.value_counts()

# for ages that are over 100, we will set them to the mean of ages
df.loc[(df.age == 110)|(df.age == 109), 'age'] = round(df[df.age < 100].age.mean(),0)

plt.hist(df.age, color="#5e3a98", bins=np.arange(15, 71, step=5))
plt.xlabel("age")
plt.show()

### ===================================status=================================== ###
# create dummies: available (1=yes, 0=no), relationship (1=in one, 0=not in one), married (1=yes, 0=no)
df['status'] = np.where(df['status'].str.contains('single'), 'available', df['status'])
df = pd.concat([pd.get_dummies(df.status, prefix='status', prefix_sep='_'), df], axis = 1)
# drop unknown since there are only a few people that fall into this category
df.drop("status_unknown", axis="columns", inplace=True)

df.status.value_counts()

### ===================================sex=================================== ###
# with make or female
df = pd.concat([pd.get_dummies(df.sex, prefix='sex', prefix_sep='_'), df], axis = 1)
# remove female column to reduce redundancy
df.drop("sex_f", axis="columns", inplace=True)

df.sex.value_counts()

### ===================================orientation=================================== ###
# either straight, gay, or bisexual
df = pd.concat([pd.get_dummies(df.orientation, prefix='orientation', prefix_sep='_'), df], axis = 1)

df.drop("orientation_bisexual", axis="columns", inplace=True)

df.orientation.value_counts()

### ===================================body_type=================================== ###
df.body_type.fillna("unknown", inplace=True)

# group body types 
df['body_type'] = np.where(df['body_type'].str.contains('athletic|fit|jacked'), 'fit', df['body_type'])
df['body_type'] = np.where(df['body_type'].str.contains('thin|skinny'), 'thin', df['body_type'])
df['body_type'] = np.where(df['body_type'].str.contains('a little extra|curvy|full figured|overweight'), 'above average', df['body_type'])
df['body_type'] = np.where(df['body_type'].str.contains('rather not say|used up'), 'unknown', df['body_type'])

# generate dummies
df = pd.concat([pd.get_dummies(df.body_type, prefix='body_type', prefix_sep='_'), df], axis = 1)
df.drop("body_type_unknown", axis="columns", inplace=True)
df.body_type.value_counts()

### ===================================diet=================================== ###
df.diet.fillna("no restriction", inplace=True)

# create dummy variable: strict (1=strictly following a diet)
df['strict'] = 0
df.loc[df.diet.str.contains('strictly'), 'strict'] = 1
df.loc[df.diet.str.len()==1, 'strict'] = 1

# group diets
df['diet'] = np.where(df['diet'].str.contains('strictly anything|mostly other|anything|mostly anything|strictly other|other'), 'no restriction', df['diet'])
df.loc[df.diet=='no restriction', 'strict'] = 0
df['diet'] = np.where(df['diet'].str.contains('mostly vegetarian|strictly vegan|strictly vegetarian|mostly vegan|vegan|vegetarian'), 'veggie', df['diet'])
df['diet'] = np.where(df['diet'].str.contains('mostly kosher|strictly kosher|kosher'), 'kosher', df['diet'])
df['diet'] = np.where(df['diet'].str.contains('mostly halal|strictly halal|halal'), 'halal', df['diet'])

# generate dummies
df = pd.concat([pd.get_dummies(df.diet, prefix='diet', prefix_sep='_'), df], axis = 1)
df.drop("diet_no restriction", axis="columns", inplace=True)
df.diet.value_counts()

# how many people are strictly following each diet?
df.groupby("diet")['strict'].mean().sort_values(ascending=False)

diet = df.groupby("diet")['strict'].mean().sort_values(ascending=False).plot(kind='bar', color="#5e3a98")
plt.xticks(np.arange(4), ['halal', 'veggie', 'kosher', 'no restriction'], rotation=45)
plt.title("% of followers that are 'strictly' following the diet")
# for p in diet.patches:
#     diet.annotate(str(round(p.get_height(), 3)), (p.get_x()*1, p.get_height()*1))
plt.show()

### ===================================drinks=================================== ###
df.drinks.fillna("socially", inplace=True)

df.drinks.value_counts().plot(kind='bar', color="#5e3a98")
plt.xticks(rotation=45)
plt.title("drinking habits")
plt.show()

df['drinks'] = np.where(df['drinks'].str.contains('not at all'), '0', df['drinks'])
df['drinks'] = np.where(df['drinks'].str.contains('rarely'), '1', df['drinks'])
df['drinks'] = np.where(df['drinks'].str.contains('socially'), '2', df['drinks'])
df['drinks'] = np.where(df['drinks'].str.contains('often'), '3', df['drinks'])
df['drinks'] = np.where(df['drinks'].str.contains('very often'), '4', df['drinks'])
df['drinks'] = np.where(df['drinks'].str.contains('desperately'), '5', df['drinks'])

df.drinks = df.drinks.astype('int64')

### ===================================drugs=================================== ###
df.drugs.fillna("maybe", inplace=True)

df.drugs.value_counts().plot(kind='bar', color="#5e3a98")
plt.xticks(rotation=45)
plt.show()

df['drugs'] = np.where(df['drugs'].str.contains('never'), '0', df['drugs'])
df['drugs'] = np.where(df['drugs'].str.contains('maybe'), '1', df['drugs'])
df['drugs'] = np.where(df['drugs'].str.contains('sometimes'), '2', df['drugs'])
df['drugs'] = np.where(df['drugs'].str.contains('often'), '3', df['drugs'])

df.drugs = df.drugs.astype('int64')

### ===================================education=================================== ###
df.education.fillna("0", inplace=True)

# if the person dropped out of somwhere, make it a 1, otherwise, a 0
df['dropped_out'] = np.where(df['education'].str.contains('dropped out'), 1, 0)

# if the person graduated from somwhere, make it a 1, otherwise, a 0
df['graduated'] = np.where(df['education'].str.contains('graduated'), 1, 0)

# verify
df[df.education.str.contains('dropped out')][['education', 'dropped_out', 'graduated']].head(2)

# we can use conditionals to rank the level of education by the time invested on scale from 0 - 6
df['recent_education_level'] = np.where(df['education'].str.contains('two-year'), '1', '0')
df['recent_education_level'] = np.where(df['education'].str.contains('college/university'), '2', df['recent_education_level'])
df['recent_education_level'] = np.where(df['education'].str.contains('masters program'), '3', df['recent_education_level'])
df['recent_education_level'] = np.where(df['education'].str.contains('med school') | df['education'].str.contains('law school'), '4', df['recent_education_level'])
df['recent_education_level'] = np.where(df['education'].str.contains('ph.d'), '5', df['recent_education_level'])

# verify
df[['education', 'dropped_out', 'graduated', 'recent_education_level']].head()

df.recent_education_level.value_counts().plot(kind='bar', color="#5e3a98")
plt.xticks(np.arange(6), ["college/university", "masters", "high school", "two-year", "ph.d", "law/med school"],rotation=45)
plt.show()

df.recent_education_level = df.recent_education_level.astype('int64')

### ===================================ethnicity=================================== ###
df.ethnicity.fillna("other", inplace=True)
df.ethnicity = df.ethnicity.str.split(', ')

# ethnicity distribution
df['eth_num'] = df.ethnicity.str.len()
df["ethnicity2"] = "race"
df.loc[df.eth_num<2, "ethnicity2"] = df.ethnicity.str[0]
df.loc[df.eth_num==2, "ethnicity2"] = "biracial"
df.loc[df.eth_num>2, "ethnicity2"] = "multiracial"
df.loc[df.eth_num>2, "eth_num"] = 3

# plot distribution
df.ethnicity2.value_counts().plot(kind='bar', color="#5e3a98")
plt.xticks(rotation=45)
plt.show()

# how many unique races are there? --9
race = pd.DataFrame(df.explode("ethnicity").ethnicity)
# race.reset_index(inplace=True)
race.ethnicity.nunique()

df[df.ethnicity.str.len()==1].shape[0] # 53087 single race
df[df.ethnicity.str.len()==2].shape[0] # 5412 two races

# create dummy variable: 1=more than 3 races
df['mix'] = 0
df.loc[df.ethnicity.str.len()>2, 'mix'] = 1



### ===================================height=================================== ###
mean_height = round(np.mean(df.loc[(df.height>=36) & (df.height<=84)].height), 0)
df.loc[(df.height<36)|(df.height>84)|df.height.isna(), 'height'] = int(mean_height)

df.height.plot(kind='hist', color="#5e3a98")
plt.xticks(np.arange(36, 96, step=6), [str(i) for i in np.arange(3, 8, step=0.5)])
plt.xlabel("height (ft.)")
plt.show()

### ===================================income=================================== ###
#df.income.isnull().sum()
df.income.value_counts()

df.income.value_counts().iloc[1:].sort_index().plot(kind='bar', color="#5e3a98")
plt.show()

### ===================================job=================================== ###
# categorized by ISCO skill levels(International Standard Classification of Occupations)
       
# Unemployed or other: -1
# 1 Military: 'military'
# 2 Service and sales workers: 'sales / marketing / biz dev','transportation','hospitality / travel',
# 3 Clerical support workers:  'clerical / administrative'
# 4 Technicians and associate professionals: 'artistic / musical / writer', 'entertainment / media', 'construction / craftsmanship',
# 5 Professionals:   'computer / hardware / software', 'banking / financial / real estate','medicine / health','science / tech / engineering', 
#                 'law / legal services', 'education / academia',
# 6 Managers: 'executive / management','political / government'

df.job.isnull().sum()
df.job.fillna('rather not say',inplace=True)

df.job = np.where(df.job.str.contains('other|student|unemployed|retired|rather'), '-1', df.job)
df.job = np.where(df.job.str.contains('military'), '1', df.job)
df.job = np.where(df.job.str.contains('sales|transportation|hospitality'), '2', df.job)
df.job = np.where(df.job.str.contains('clerical'), '3', df.job)
df.job = np.where(df.job.str.contains('artistic|media|construction'), '4', df.job)
df.job = np.where(df.job.str.contains('computer|banking|medicine|science|law|education'), '5', df.job)
df.job = np.where(df.job.str.contains('executive|political'), '6', df.job)
df.job = df.job.astype(int)
df.job.nunique()

df.groupby('job').size().plot(kind='bar', color="#5e3a98")
plt.show()

### ===================================last_online=================================== ###
# df.last_online.isnull().sum()
df.last_online.value_counts()

l = []
for i in df.last_online:
    j = i.split("-")
    l.append(int(j[0]+j[1]+j[2]))
df['last_online_date'] = pd.to_datetime(l,format='%Y%m%d')
sns.histplot(df,x='last_online_date',color="#5e3a98")
plt.show()

### ===================================location=================================== ###
west = 'Wyoming|Washington|Utah|Oregon|Alaska|Arizona|California|Colorado|Hawaii|Idaho|Montana|Nevada|New Mexico'
central = 'Wisconsin|Texas|Tennessee|South Dakota|Oklahoma|Alabama|Arkansas|Illinois|Indiana|North Dakota|Iowa|Kansas|Nebraska|Louisiana|Minnesota|Mississippi|Missouri'
northeast = 'Vermont|Pennsylvania|Rhode Island|Ohio|Connecticut|Maine|Massachusetts|Michigan|New Hampshire|New Jersey|New York'
southeast = 'West Virginia|Virginia|Delaware|Florida|Georgia|Kentucky|Maryland|North Carolina|South Carolina'
international = ''

df['region'] = 'international'

df.loc[df['location'].str.contains(west.lower()), 'region'] = "west"
df.loc[df['location'].str.contains(central.lower()), 'region'] = "central"
df.loc[df['location'].str.contains(northeast.lower()), 'region'] = "northeast"
df.loc[df['location'].str.contains(southeast.lower()), 'region'] = "southeast"

df = pd.concat([pd.get_dummies(df.region, prefix='region', prefix_sep='_'), df], axis = 1)
df.drop("region_international", axis='columns', inplace=True)

df.region.value_counts()

### ===================================offspring=================================== ###
# df.offspring.value_counts().sort_index()
df.offspring.fillna("unknown", inplace=True)

# create 2 variables: have kids (1=yes, 0=no), want kids (1=yes, 0=neutral, -1=no)
df["havekids"] = 1
df["wantkids"] = 1
df.loc[df.offspring.str.contains(r"doesn't have"), "havekids"] = 0
df.loc[df.offspring.str.contains(r"doesn't want"), "wantkids"] = -1
df.loc[(df.offspring=="doesn't want kids")|(df.offspring=="might want kids")|(df.offspring=="wants kids")|(df.offspring=="unknown"), "havekids"] = 0
df.loc[(df.offspring=="doesn't have kids")|(df.offspring=="has a kid")|(df.offspring=="has kids")|(df.offspring=="unknown"), "wantkids"] = 0
# df.groupby("offspring")[["havekids","wantkids"]].mean()

# heatmap
kids = pd.crosstab(df.havekids, df.wantkids)
kids.sort_index(ascending=False, inplace=True)
sns.heatmap(kids, cmap="Purples", annot=True, fmt='d'
            ,xticklabels=["no","neutral","yes"]
            ,yticklabels=["yes", "no"]
            )
plt.xlabel("want kids")
plt.ylabel("have kids")
plt.show()

### ===================================pets=================================== ###
# df.pets.value_counts().sort_values()
df.pets.fillna("unknown", inplace=True)

# create 2 variables: dogs, cats (1=like, 0=neutral, -1=dislike)
df["dogs"] = 1
df["cats"] = 1
df.loc[df.pets.str.contains("dislikes dogs"), "dogs"] = -1
df.loc[df.pets.str.contains("dislikes cats"), "cats"] = -1
df.loc[df.pets.str.contains("has dogs"), "dogs"] = 0
df.loc[df.pets.str.contains("has cats"), "cats"] = 0
df.loc[(df.pets=="has dogs")|(df.pets=="likes dogs")|(df.pets=="dislikes dogs")|(df.pets=="unknown"), "cats"] = 0
df.loc[(df.pets=="has cats")|(df.pets=="likes cats")|(df.pets=="dislikes cats")|(df.pets=="unknown"), "dogs"] = 0
# df.groupby("pets")[["dogs","cats"]].mean()

# heatmap
pets = pd.crosstab(df.dogs, df.cats)
pets.sort_index(ascending=False, inplace=True)
sns.heatmap(pets, cmap="Purples", annot=True, fmt='d', 
            xticklabels=["dislike","neutral","like"],
            yticklabels=["like", "neutral", "dislike"])
plt.show()

### ===================================religion=================================== ###
# df.religion.value_counts().sort_index()
df.religion.fillna("other", inplace=True)

# create 1 variable: serious (1=yes, 0=neutral, -1=no)
df["serious"] = 0
df.loc[df.religion.str.contains("very|somewhat"), "serious"] = 1
df.loc[df.religion.str.contains("laughing"), "serious"] = -1
df.religion = df.religion.str.split().str[0]
# df.groupby("religion")["serious"].mean()

# seriousness by religion
plt.plot(df.groupby("religion")["serious"].mean(), marker='o', color="#5e3a98")
plt.xticks(rotation=45)
plt.title("how seriously does a user take religion?")
plt.ylabel("average seriousness")
plt.show()

### ===================================sign=================================== ###
# df.sign.value_counts().sort_index()
df.sign.fillna("unknown", inplace=True)

# create 1 variable: importance (1=yes, 0=no)
df["importance"] = 0
df.loc[df.sign.str.contains("a lot"), "importance"] = 1
df.sign = df.sign.str.split().str[0]
# df.groupby("sign")["importance"].mean()

# importance by sign
plt.plot(df.groupby("sign")["importance"].mean(), marker='o', color="#5e3a98")
plt.xticks(rotation=45)
plt.title("how much do signs matter? (on the scale of 0 to 1)")
plt.show()

### ===================================smokes=================================== ###
# df.smokes.value_counts()
df.smokes.fillna("maybe", inplace=True)

# create 3 types: yes, no, unknown
# df.smokes = df.smokes.str.replace(r"^s[\w\s]*|^w[\w\s]*|^t[\w\s]*", "yes")
df.loc[df.smokes=='no', 'smokes'] = '0'
df.loc[df.smokes=='trying to quit', 'smokes'] = '1'
df.loc[df.smokes.str.contains('seems important to say'), 'smokes'] = '1'
df.loc[df.smokes=='maybe', 'smokes'] = '1'
df.loc[df.smokes=='when drinking', 'smokes'] = '2'
df.loc[df.smokes=='sometimes', 'smokes'] = '3'
df.loc[df.smokes=='yes', 'smokes'] = '4'
df.smokes = df.smokes.astype(int)

# plot
df.smokes.value_counts().plot(kind="bar", color="#5e3a98")
plt.xticks(rotation=0)
plt.xlabel("smokes")
plt.show()

### ===================================speaks=================================== ###
# df.speaks.value_counts().sort_index()
df.speaks.fillna("english", inplace=True)
df.speaks = df.speaks.str.split(', ')

lng = pd.DataFrame(df.explode("speaks").speaks)
lng.reset_index(inplace=True)
lng.drop(lng[lng.speaks.str.contains("poorly")].index, inplace=True)
lng.speaks = lng.speaks.str.split().str[0]

missing = sorted(set(range(df.shape[0]))-set(lng["index"]))
for i in missing:
    lng = pd.concat([lng, pd.DataFrame({"index":[i], "speaks":["english"]})])

lng = lng.groupby("index").agg({"speaks": lambda x: x.tolist()})
df.speaks = lng.speaks

# onehot dataframe of languages
encoder = TransactionEncoder().fit(df.speaks)
onehot = encoder.transform(df.speaks)
onehot = pd.DataFrame(onehot, columns = encoder.columns_)
onehot.mean().sample(10)

# 10 most popular languages
top10 = onehot.mean().sort_values(ascending=False).iloc[:10]
plt.plot(top10, marker='o', color="#5e3a98")
plt.xticks(rotation=45)
plt.ylabel("% of users")
plt.title("top 10 most popular languages")
plt.show()

# 10 most popular languages besides English and other
top12 = onehot.mean().sort_values(ascending=False).iloc[1:12]
top12.drop("other", inplace=True)
plt.plot(top12, marker='o', color="#5e3a98")
plt.xticks(rotation=45)
plt.ylabel("% of users")
plt.title("top 10 most popular languages (besides English and other)")
plt.show()

# create 1 variable: languages (how many does one speak)
df["languages"] = df.speaks.str.len()
# df.languages.value_counts()
df.languages.value_counts().plot(kind="bar", color="#5e3a98")
plt.xticks(rotation=0)
plt.title("how many languages does a user speak?")
plt.show()

eth_lan = np.round(pd.crosstab(df.languages, df.ethnicity2)/pd.crosstab(df.ethnicity2, df.languages).sum(axis=1), 3).T
sns.heatmap(eth_lan, cmap='Purples', annot=True, fmt='g')
plt.title('% of users in each ethnic group that speak 1-5 languages')
plt.ylabel("")
plt.show()

### ================================================================================= ###
### ================================LANGUAGE ANALYSIS================================ ###

# setup blank columns for language
df['essay0_language'] = pd.Series()
df['essay1_language'] = pd.Series()
df['essay2_language'] = pd.Series()
df['essay3_language'] = pd.Series()
df['essay4_language'] = pd.Series()
df['essay5_language'] = pd.Series()
df['essay6_language'] = pd.Series()
df['essay7_language'] = pd.Series()
df['essay8_language'] = pd.Series()
df['essay9_language'] = pd.Series()

df = df.reset_index()
pd.options.mode.chained_assignment = None

# check languages for every essay
for (columnName, columnData) in df.iteritems():
    print(columnName)
    if ('essay' in columnName) & ('_language' not in columnName):
        for i, item in enumerate(df[columnName]):
            try:
                result = str(detect_langs(item)[0]).partition(':')
                if float(result[2]) >= 0.95:
                    lang = result[0]
                else:
                    lang = 'en'
                df[columnName + '_language'].loc[i] =  lang
            except:
                df[columnName + '_language'].loc[i] =  np.NaN


# for any values that are NA, or were marked with an error by the language processing will be
# set to "." so that in the next step, the sentiment for these will be 0
for (columnName, columnData) in df.iteritems():
    if ('essay' in columnName) & ('_language' not in columnName):
        temp_col = columnName + '_language'
        df.loc[df[temp_col].isna(), columnName] = '.'


### ================================================================================= ###
### =================================ENTITY ANALYSIS================================= ###
# we would like to analyze often users mention their hobbies or interests in their profiles
hobbies = pd.read_csv("C:\\Users\\pmank\\Dropbox\\BU\\BA820\\hobbies.csv")

nlp = English()
spacy.cli.download("en_core_web_md")
nlp = spacy.load('en_core_web_md')

def least_frequent(ents):
    temp = []
    if len(ents) > 0:
        for item in ents:
            temp.append(item[1])
        return min(set(temp), key = ents.count) 
    else:
        return "none"

def most_frequent(ents):
    temp = []
    if len(ents) > 0:
        for item in ents:
            temp.append(item[1])
        return max(set(temp), key = ents.count) 
    else:
        return "none"


def iscardinal(ents):
    temp = []
    if len(ents) > 0:
        for item in ents:
            temp.append(item[1])
    if 'CARDINAL' in temp:
        return 1
    else:
        return 0

def isnord(ents):
    temp = []
    if len(ents) > 0:
        for item in ents:
            temp.append(item[1])
    if 'NORD' in temp:
        return 1
    else:
        return 0

def isperson(ents):
    temp = []
    if len(ents) > 0:
        for item in ents:
            temp.append(item[1])
    if 'PERSON' in temp:
        return 1
    else:
        return 0

def istime(ents):
    temp = []
    if len(ents) > 0:
        for item in ents:
            temp.append(item[1])
    if 'TIME' in temp:
        return 1
    else:
        return 0

def isorg(ents):
    temp = []
    if len(ents) > 0:
        for item in ents:
            temp.append(item[1])
    if 'ORG' in temp:
        return 1
    else:
        return 0

def isordinal(ents):
    temp = []
    if len(ents) > 0:
        for item in ents:
            temp.append(item[1])
    if 'ORDINAL' in temp:
        return 1
    else:
        return 0

def isgpe(ents):
    temp = []
    if len(ents) > 0:
        for item in ents:
            temp.append(item[1])
    if 'GPE' in temp:
        return 1
    else:
        return 0

# add the lemma but not for two word hobbies because this becomes not meaningful
def hobbies_def(text):
    doc = nlp(text)
    for token in doc:
        if len(doc) > 1:
            return np.NaN
        else:
            return token.lemma_

# find the entities in each document for each user
def spacy_ent(text):
    ents_list = []
    doc = nlp(text)
    for e in doc.ents:
        ents_list.append((e.text, e.label_))
    return ents_list

# add the hobbies found to a new column for each user
def hobbies_find(text):
    temp = []
    for i, (hobby, hobby_lemma) in enumerate(zip(hobbies['HOBBIES'].values, hobbies['lemma'].values)):
        if (isinstance(hobby, str)) & (isinstance(hobby_lemma, str)):
            if (hobby in text) | (hobby_lemma in text):
                temp.append(hobby)
    return temp

# column with text from all 10 essay columns concatonated 
cols = ['essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay8']
df['essay_combined'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

df['essay_ents'] = df.essay_combined.apply(spacy_ent)

# least frequent and most frequent entities
df['ent_least_freq'] = df['essay_ents'].apply(least_frequent)
df['ent_most_freq'] = df['essay_ents'].apply(most_frequent)

# create dummy variables for the occurance of certain entities
df['ent_cardinal'] = df['essay_ents'].apply(iscardinal)
df['ent_nord'] = df['essay_ents'].apply(isnord)
df['ent_person'] = df['essay_ents'].apply(isperson)
df['ent_time'] = df['essay_ents'].apply(istime)
df['ent_org'] = df['essay_ents'].apply(isorg)
df['ent_ordinal'] = df['essay_ents'].apply(isordinal)
df['ent_gpe'] = df['essay_ents'].apply(isgpe)

# mkae hobbies lowercase
hobbies['HOBBIES'] = hobbies['HOBBIES'].str.lower()

# find the lemma of each hobby
hobbies['lemma'] = hobbies['HOBBIES'].apply(hobbies_def)

# find the hobbies mentioned by each user
df['hobbie_count'] = df['essay_combined'].apply(hobbies_find)


### ================================================================================= ###
### ====================================SENTIMENT==================================== ###
from textblob import TextBlob
# df.info()
# df['essay0'].replace({np.nan:'neutral'},inplace=True)
dic = {}
sen_col = []
sub_col =[]
all_text = []
for i in range(10):
    col_n = 'essay'+ str(i)
    data = df[col_n].replace({np.nan:'neutral'})
    parsed = []
    senti = []
    subject = []
    for j in data.values:
        parsed_text = TextBlob(j)
        senti.append(parsed_text.sentiment.polarity)
        subject.append(parsed_text.sentiment.subjectivity)
        parsed.append(parsed_text)
        all_text.append(j)
    dic[col_n+'sen'] = senti
    sen_col.append(col_n+'sen')
    dic[col_n+'sub'] = subject
    sub_col.append(col_n+'sub')
essay_df = pd.DataFrame(dic)
# check essay descriptative statistics
# essay_df.describe()
df['sentiment_avg'] = essay_df[sen_col].apply(np.mean,axis=1)
df['subjectivity_avg'] = essay_df[sub_col].apply(np.mean,axis=1)


### ================================================================================= ###
### ===================================PCA K-Means=================================== ###
df_num = df.select_dtypes('number')
df_num.info()
ds = StandardScaler().fit_transform(df_num)
data_scaled = pd.DataFrame(ds,columns=df_num.columns)
data_scaled.describe().T

pca = PCA(0.95)
pcs = pca.fit_transform(ds)
varexp = pca.explained_variance_ratio_
pca.n_components_

plt.title("Cumulative Explained Variance")
plt.plot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.95)
plt.show()


columns_name = []
for i in range(pca.n_components_):
    columns_name.append('pca20_'+ str(i+1))
pca_df = pd.DataFrame(pcs,columns=columns_name)

krange_pca = range(2,20)
inertia_pca =[]
silo_pca = []
for k in krange_pca:
    kmodel = KMeans(k)
    k_labs = kmodel.fit_predict(pca_df)
    inertia_pca.append(kmodel.inertia_)
    silo_pca.append(silhouette_score(pca_df,k_labs))

#Show Inertia plot
sns.lineplot(krange_pca,inertia_pca)
plt.title('k value and inertia')
plt.show()

#Show Silhouette plot
sns.lineplot(krange_pca, silo_pca)
plt.title('k value and silhouette score')
plt.show()

### ================================================================================= ###
### ==================================t-SNE K-Means================================== ###
tsne = TSNE()
tsne.fit(data_scaled)
te = tsne.embedding_
tsne_df = pd.DataFrame(te,columns=['e1','e2'])

krange_8 = range(2,8)
inertia_8 =[]
silo_8 = []
for k in krange_8:
    kmodel = KMeans(k)
    k_labs = kmodel.fit_predict(tsne_df)
    inertia_8.append(kmodel.inertia_)
    silo_8.append(silhouette_score(tsne_df,k_labs))

#Show Inertia plot
sns.lineplot(krange_8,inertia_8)
plt.title('k value and inertia')
plt.show()

#Show Silhouette plot
sns.lineplot(krange_8, silo_8)
plt.title('k value and silhouette score')
plt.show()

## Use k = 3 to produce clusters 
k_tsne = KMeans(3)
k_tsne_labs = k_tsne.fit_predict(tsne_df)
# silhouette score
skplot.metrics.plot_silhouette(tsne_df, k_tsne_labs, title="KMeans-3-tsne", figsize=(15,5))
plt.show()

df['tsne_labels'] = k_tsne_labs
df.groupby('tsne_labels').mean().T


### ================================================================================= ###
### =========================t-SNE K-Means (specific groups)========================= ###
def input_stats(df=None):
    '''Input dataframe; Enter parameter required; Output Descriptive statistics for each groups
        ;Using Tsne and KMeans as methods'''
    df_num = df.select_dtypes('number')
    ds = StandardScaler().fit_transform(df_num)
    data_scaled = pd.DataFrame(ds,columns=df_num.columns)
    print('Numeric shape of dataframe is: ',df_num.shape)

    tsne = TSNE()
    tsne.fit(data_scaled)
    te = tsne.embedding_
    tsne_df = pd.DataFrame(te,columns=['e1','e2'])
    
    s,e = int(input('k range start:')), int(input('k range end:'))

    krange = range(s,e+1)
    inertia =[]
    silo = []
    for k in krange:
        kmodel = KMeans(k)
        k_labs = kmodel.fit_predict(tsne_df)
        inertia.append(kmodel.inertia_)
        silo.append(silhouette_score(tsne_df,k_labs))
    
    print('Be advice! You will have to choose k from below two graphs for next process!')

    sns.lineplot(krange,inertia)
    plt.title('k value and inertia')
    plt.show()

    sns.lineplot(krange, silo)
    plt.title('k value and silhouette score')
    plt.show()

    dfcopy = df.copy()
    k = int(input('input optimal k:'))
    km = KMeans(k)
    k_labs = km.fit_predict(tsne_df)
    dfcopy['kmeans_labels'] = k_labs

    return (dfcopy.groupby('kmeans_labels').mean().T)

### ==================================================================== ###
### ============================female group============================ ###
col = ['diet_halal', 'diet_kosher', 'diet_veggie', 'body_type_above average',
       'body_type_average', 'body_type_fit', 'body_type_thin',
       'orientation_gay', 'orientation_straight', 'sex_m', 'status_available',
       'status_married', 'status_seeing someone', 'age',
       'drinks', 'drugs', 
       'height', 'income', 'job',
    #    'religion', 'sign', 
       'smokes',
       'strict', 'dropped_out', 'graduated',
       'recent_education_level', 'eth_num',
       'havekids', 'wantkids', 'dogs', 'cats',
    #    'serious', 'importance', 
       'languages', 'sentiment_avg', 'subjectivity_avg']

df2 = df.loc[:, col]
female = df2[df2['sex_m']==0]
tmp_f = input_stats(female)
tmp_f

# female_heter = female[female['orientation_straight']==1]
# female_homo = female[female['orientation_gay']==1]
# female_bi = female[(female['orientation_straight']==0)&(female['orientation_gay']==0)]


### ============================male group============================ ###
male = df2[df2['sex_m']==1]
tmp_m = input_stats(male)
tmp_m
# male_heter = male[male['orientation_straight']==1]
# male_homo = male[male['orientation_gay']==1]
# male_bi = male[(male['orientation_straight']==0)&(male['orientation_gay']==0)]

### =============================below 30============================= ###
ageb30 = df2.loc[df2['age']< 30,:]
tmp_b30 = input_stats(ageb30)
tmp_b30

### =============================above 30============================= ###
agea30 = df2.loc[df2['age']> 30,:]
tmp_a30 = input_stats(agea30)
tmp_a30




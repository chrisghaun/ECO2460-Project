# Written by James Cabral and Chris Haun
# This program processes the data from the web scrape and prepares it for the NLP steps

#NOTE: Some of the implementation follows the steps from the replication code of Bisbee et al. (2021)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import LdaModel, CoherenceModel
from gensim import models
from gensim.corpora import Dictionary
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import unicodedata


os.chdir(r"C:\Users\jcabr\OneDrive - University of Toronto\Coursework\Winter\ECO2460\Empirical Project\Data")

text_data = pd.read_csv("text_forNLP.csv")

########################Pre-Processing Steps for text

# cleaning the strings. Authors get rid of anything that is not a letter
# get rid of double spaces, appostrophes
#everything lower case
def clean_function(doc):
    
    #dealing with accents: this will replace é with e instead of breaking up the word
    doc = unicodedata.normalize('NFKD', doc)
    doc = ''.join([c for c in doc if not unicodedata.combining(c)])

    doc = doc.lower()
    doc = doc.replace("'", "")
    doc = re.sub(r'[^a-zA-Z]', ' ', doc)
    doc = re.sub(r'\s+', ' ', doc)
    return doc


#apply the function to the data
#text_data['cleaned_text_1'] = text_data['text_clean'].apply(clean_function)

#lemmatizing: Takes about 10 minutes for small sample

# Use POS tags to handle verbs and adjectives
import swifter
def get_wordnet_pos(nltk_tag):
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(nltk_tag[0].upper(), wordnet.NOUN)

lemma = WordNetLemmatizer()
def lemmatize_text(text):
    words = word_tokenize(text) 
    pos_tags = pos_tag(words) 
    return " ".join(lemma.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags)

text_data['cleaned_text_2'] = text_data['cleaned_text_1'].swifter.apply(lemmatize_text)
text_data.drop(columns=['cleaned_text_1'], inplace=True)

####
#SAVE TO CSV

text_data.to_csv('text_forNLP_lemmatized.csv', index=False)

#re-import data with lemmatization. The data have been cleaned and lemmatized
text_data = pd.read_csv('text_forNLP_lemmatized.csv')


#define a set of stopwords and remove them from the data
stop1 = set(ENGLISH_STOP_WORDS)
stop1 = {word.replace("'", "") for word in stop1}

stop2 = set(stopwords.words('english'))
stop2 = {word.replace("'", "") for word in stop2}

stop3 = {'u', 'th', 'go', 'moreover', 'like', 'since', 'mine', 'mr', 'mrs', 'nd', 'rd', 
         'aaa', 'aga', 'ai', 'al', 'ali', 'ar', 'au', 'use', 'go',
         'talk', 'say', 'go', 'hon', 'oh', 'know', 'thank',
         'ax', 'ba', 'c', 'hello', 'bounjour', 'call', 'upon', 'today',
         'stand', 'pleasure', 'thankful', 'merci', 'm', 'chair', 'member',
         'order', 'house', 'proceed', 'pursuant', 'clerk', 'congratulation', 
         'speaker', 'address', 'congratulate', 'pleased', 'need', 'privilege',
         'nice', 'want', 'yesterday', 'honourable', 'motion', 'madam', 'make',
         'day', 'floor', 'declare', 'think'}

#extra words found in the LDA topics
stop4 = {'thing', 'come', 'look', 'time', 'way', 'lot', 'really', 'year', 
         'sure', 'place', 'happen', 'hear', 'try', 'ask', 'let', 'bring'}

all_stopwords = stop1 | stop2 | stop3 | stop4

text_data['cleaned_text_3'] = text_data['cleaned_text_2'].apply(
    lambda doc: [val for val in doc.split() if val not in all_stopwords]
)
text_data['cleaned_text_3'] = text_data['cleaned_text_3'].apply(" ".join)

#text_data_backup = text_data

##########################################
# Sentiment Analysis: Harvard Vocabulary #
##########################################

########################Importing the Lexicon
#Three core sentiment measures:

#harvard lexicon
#downloaded from: https://inquirer.sites.fas.harvard.edu/spreadsheet_guide.htm
lexicon = pd.read_excel("inquirerbasic.xls")

hostile_lexicon = lexicon['Entry'][lexicon["Hostile"].notna()]

hostile_lexicon = hostile_lexicon.apply(lambda text: "" if isinstance(text, bool) else text)


##################################
#cleaning up and lemmatizing the lexicon for consistency with the text
hostile_lexicon = hostile_lexicon.apply(clean_function)

hostile_lexicon = hostile_lexicon.apply(lambda text: " ".join(lemma.lemmatize(word, get_wordnet_pos(word)) for word in word_tokenize(text)))
hostile_lexicon = set(hostile_lexicon)

#take out some words that may not be relevant for our purpose
hostile_lexicon = hostile_lexicon.difference({'floor', 'fun', 'keep', 'fine', 'pan', 'shell', 'stop'
                                              'make', 'serve', 'raise', 'wheek', 'no', 'vie',
                                              'root', 'ride', 'belt', 'tax', 'arm', 'bit', 'try',
                                              'even', 'time', 'point', 'turn', 'tire', 'mine',
                                              'get', 'pas', 'wait', 'act', 'let', 'bar', 'bit',
                                              'time', 'run', 'put', 'question', 'comment', 'whip',
                                              'opposition'})

#Approach #1: Hostile words / total words per utterance
# note: I am taking the total words for the utterance before any cleaning was done
text_data['original_length'] = text_data['text_clean'].apply(lambda text: len(text.split()))

text_data['hostile_word_count'] = text_data['cleaned_text_3'].apply(lambda text: sum(1 for word in text.split() if word in hostile_lexicon))
text_data["SENT_Hostile"] = text_data['hostile_word_count']/text_data['original_length']

########
# Filter for the LDA model
#take out 
text_data_lda = text_data[(text_data['party'] != "none") & text_data['party'].notna()]
text_data_lda['cleaned_text_tokens'] = text_data_lda['cleaned_text_3'].apply(word_tokenize)

#################################################
# Tuning the Number of Topics for the LDA Model #
#################################################

from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
np.random.seed(2460)

#################
# Looking for the optimal number of topics
# Check both coherence of topics for each K value

#split the data: Take 2k randomly from each year
text_data_lda['date'] = pd.to_datetime(text_data_lda['date'])
text_data_lda['year'] = text_data_lda['date'].dt.year
text_data_testing = text_data_lda.groupby('year', group_keys=False).apply(lambda x: x.sample(n=2000, random_state=2460))

#create the dictionary and corpus for the test set
DTM_test = Dictionary(text_data_testing['cleaned_text_tokens'].tolist())
DTM_test.filter_extremes(no_below=30, no_above=0.3)

#for a manual check. Turn into a data frame
#use this to get rid of some more stopwords
DTM_df_test = pd.DataFrame(list(DTM_test.token2id.items()), columns=['token', 'id'])
DTM_df_test = DTM_df_test.sort_values(by="token").reset_index(drop=True)

corpus_test = [DTM_test.doc2bow(text) for text in text_data_testing['cleaned_text_tokens']]

k_values = [5, 10, 25, 40, 50, 60, 80, 100]
coherence_results = [] 

for k in k_values:
    print(k)
    ldamodel_final = LdaMulticore(
        corpus_test, id2word=DTM_test, num_topics=k, alpha=0.1, eta=0.01,
        iterations=500, eval_every=25, passes=5,
        chunksize=5000,
        random_state=2460, workers=4 
    )
    
    coherence_model_lda = models.CoherenceModel(
        model=ldamodel_final, 
        corpus=corpus_test,
        texts= text_data_lda['cleaned_text_tokens'],
        dictionary=DTM_test, 
        coherence='c_v',
    processes=4) 
    coherence_scores = coherence_model_lda.get_coherence_per_topic()
    
    #filter out the infinite values and get mean coherence score
    coher_scores_filtered = [score for score in coherence_scores if np.isfinite(score)]
    coherence_score = np.mean(coher_scores_filtered)
    
    #store the k value and the coherence score
    coherence_results.append({'k': k, 'coherence': coherence_score})

coherence_results_df = pd.DataFrame(coherence_results)

# Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=coherence_results_df, x='k', y='coherence', label="Coherence (cv)")
sns.lineplot(data=coherence_results_df, x='k', y='coherence')

# Labels
plt.xlabel("Number of Topics (k)")
plt.ylabel("Coherence (cv)")
plt.title("Coherence vs. Number of Topics")

plt.show()

###################
# Final LDA Model #
###################
#Use the k value with the highest coherence score

#generate the dictionary now using the full sample
DTM = Dictionary(text_data_lda['cleaned_text_tokens'].tolist())
DTM.filter_extremes(no_below=30, no_above=0.3)

#for a manual check. Turn into a data frame
#use this to get rid of some more stopwords
DTM_df = pd.DataFrame(list(DTM.token2id.items()), columns=['token', 'id'])
DTM_df = DTM_df.sort_values(by="token").reset_index(drop=True)

corpus = [DTM.doc2bow(text) for text in text_data_lda['cleaned_text_tokens']]

#load a previously generated model instead of running it again
#ldamodel_final = LdaModel.load("lda_model_2.gensim")

#Run the final model
#Using 1000 and 2500 iterations gave consistent results and took similar amount of time to run
ldamodel_final = LdaMulticore(
    corpus, id2word=DTM, num_topics=50, alpha=0.1, eta=0.01,
    iterations=2500, eval_every=25, passes=5,
    chunksize=5000,
    random_state=2460, workers=4 
)
#print the topics
print(ldamodel_final.print_topics(num_topics=5, num_words=8))

#save the model and a file with the top words per topic for future reference
ldamodel_final.save("lda_model.gensim")
topics = {
    f"topic_{i}": [word for word, _ in ldamodel_final.show_topic(i, topn=20)]
    for i in range(ldamodel_final.num_topics)
}
topics_df = pd.DataFrame.from_dict(topics, orient="index", columns=[f"word_{i+1}" for i in range(20)])
topics_df.to_csv('Topics_words.csv', index=False)

#visualize the model using pyLDAvis
#Code obtained directly from: https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know
import pyLDAvis
import pyLDAvis.gensim

pyLDAvis.enable_notebook()
p = pyLDAvis.gensim.prepare(ldamodel_final, corpus, DTM)
pyLDAvis.display(p)
pyLDAvis.save_html(p, 'lda_visualization.html')

#Now, create a distribution of the topics for each utterance
#this is to get the data into the proper format
topic_dist = [
    #loop through each of the utterances
    #get the probability of each topic for each document (utterance)
    [theta for _, theta in ldamodel_final.get_document_topics(doc, minimum_probability=0)]
    for doc in corpus]

#for merging: add an id and convert into long form
#NOTE: I manually checked that the topic probabilities line up with the content of the speeches
#e.g. the utterances with the highest topic probabiliy for the "seniors" topic were all talking about pension, etc.
topic_dist = pd.DataFrame(topic_dist)
topic_dist['id'] = range(1, len(topic_dist) + 1)
text_data_lda['id'] = range(1, len(text_data_lda) + 1)

text_data_lda = pd.merge(text_data_lda, topic_dist, how='left', on='id')

text_data_lda = text_data_lda.rename(columns={col: f"topic_{col}" for col in text_data_lda.columns if str(col).isdigit()})

text_data_lda.drop(columns=['cleaned_text_tokens', 'id','hostile_word_count', 'text_clean',
                            'cleaned_text_3', 'cleaned_text_2'
                            ]).to_csv('Final_Data_Reg.csv', index=False)

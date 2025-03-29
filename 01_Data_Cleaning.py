# Written by James Cabral and Chris Haun
# This program processes the data from the web scrape and prepares it for the NLP steps

#NOTE: Some of the implementation follows the cleaning steps from Bisbee et al. (2021)

import os 
import pandas as pd 
import numpy as np

os.chdir(r"C:\Users\jcabr\OneDrive - University of Toronto\Coursework\Winter\ECO2460\Empirical Project\Data")




# import the raw data

data_raw = pd.read_csv("raw_data_test.csv")

#generate a date variable
data_raw["date"] = pd.to_datetime(data_raw["day"].astype(str) + " " +
                            data_raw["month"] + " " + 
                            data_raw["year"].astype(str), format="%d %B %Y")

#will create a new text_clean column with cleaned up version of the text
#remove the name of the speaker
#identify the first utterance in each block_ID and remove everything before :
data_raw['first_utterance'] = data_raw.groupby(["date", "Block_ID"])["Utterance_ID"].transform("min")
data_raw['first_utterance'] = (data_raw["Utterance_ID"] == data_raw["first_utterance"]).astype(int)

data_raw.loc[data_raw['first_utterance'] == 1, 'text_clean'] = data_raw.loc[data_raw['first_utterance'] == 1, 'Utterance'].str.replace(r"^.*?:\s*", "", regex=True)
data_raw['text_clean'] = data_raw['text_clean'].fillna(data_raw['Utterance'])

##############################
# Idntifying political party #
##############################

#Problem: sometimes the name has political party, sometimes it doesn't
#plan: get political party when we can, then remove everything in () from the name, then match polirical party by name

#NOTE: would cause an issue if someone switched parties

#initialize
data_raw['party'] = np.nan

partytags = ['Lib.', 'BQ', 'CPC', 'NDP', 'GP', 'Ind.'] 

for tag in partytags:
    #use this comma in case someone happens to have these characters in their name
    party_comma = ', ' + tag
    data_raw['party'] = data_raw.apply(
        lambda row: tag if party_comma in row['Speaker_Tag'] else row['party'], 
        axis=1
    )

data_raw = data_raw.sort_values(by=['date', 'Block_ID', 'Utterance_ID'], axis=0)

#now, use the Current_Speaker variable (which doesn't have the brackets) to make sure that each speaker has the same party along the whole dataset
data_raw['party'] = data_raw.groupby('Current_Speaker')['party'].transform(lambda x: x.ffill())
data_raw['party'] = data_raw.groupby('Current_Speaker')['party'].transform(lambda x: x.bfill())


#### DO A MANUAL CHECK FOR SOMEONE WHO SWITCHED PARTIES

#replace party with NA if it's the speaker or if a special case
data_raw.loc[data_raw['Speaker_Tag'] == 'The Deputy Speaker', 'party'] = 'none'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Speaker', 'party'] = 'none'
data_raw.loc[data_raw['Current_Speaker'] == 'The Assistant Deputy Speaker', 'party'] = 'none'
data_raw.loc[data_raw['Current_Speaker'] == 'The Chair', 'party'] = 'none'
data_raw.loc[data_raw['Current_Speaker'] == 'The Deputy Chair', 'party'] = 'none'
data_raw.loc[data_raw['Current_Speaker'] == 'The Presiding Officer', 'party'] = 'none'
data_raw.loc[data_raw['Current_Speaker'] == 'The Acting Speaker', 'party'] = 'none'
data_raw.loc[data_raw['Current_Speaker'] == 'The Assistant Deputy Chair', 'party'] = 'none' 
data_raw.loc[data_raw['Current_Speaker'] == 'The Clerk of the House', 'party'] = 'none' 
data_raw.loc[data_raw['Current_Speaker'] == 'The Assistant Deputy Chair', 'party'] = 'none' 

data_raw.loc[data_raw['Current_Speaker'] == 'His Excellency Volodymyr Zelenskyy', 'party'] = 'none'
data_raw.loc[data_raw['Current_Speaker'] == 'H.E. Volodymyr Zelenskyy', 'party'] = 'none'
data_raw.loc[data_raw['Current_Speaker'] == 'Mr. Speaker Rota', 'party'] = 'none'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Raymonde Gagné', 'party'] = 'none' #Speaker of the senate
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Joseph Biden, Jr.', 'party'] = 'none' 
data_raw.loc[data_raw['Current_Speaker'] == 'Her Excellency Ursula von der Leyen', 'party'] = 'none' 

data_raw.loc[data_raw['Current_Speaker'] == 'Mrs. Soraya Martinez Ferrada', 'party'] = 'none' 
data_raw.loc[data_raw['Current_Speaker'] == 'Mr. Kristian Firth', 'party'] = 'none' 


#to revisit once we have full dataset

#check to see if there are still missing parties
noparty = data_raw.loc[data_raw['party'].isna()]


#####################
# Idntifying gender #
#####################
#for speakers, I am obtaining information from: 
#https://lop.parl.ca/sites/ParlInfo/default/en_CA/People/OfficersParliament/politicalOfficersCommons/deputySpeakers

#NOTE: Code will not work if this is being run for time periods where the speaker positions change

#initialize
data_raw['gender'] = np.nan

data_raw.loc[data_raw['Speaker_Tag'].str.startswith('Mr.'), 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'].str.startswith('Mrs.'), 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'].str.startswith('Ms.'), 'gender'] = 'F'

####
#The Speaker: 
#No female speaker since Jeanne Sauve in 1980-1984
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Speaker'), 'gender'] = 'M'

####
#The Deputy Speaker
#Chris d'Entremon (still the DS), Bruce Stanton, Joseph Comartin
#https://lop.parl.ca/sites/ParlInfo/default/en_CA/People/OfficersParliament/politicalOfficersCommons/deputySpeakers
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Deputy Speaker') & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('2012-09-17')), 'gender'] = 'M'
#Denise Savoie
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Deputy Speaker') & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('2011-06-06')) & 
             (pd.to_datetime(data_raw['date']) < pd.to_datetime('2012-09-17')), 'gender'] = 'F'
#Andrew Scheer, William Blaikie, Charles Strahl
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Deputy Speaker') & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('1994-01-18')) & 
             (pd.to_datetime(data_raw['date']) < pd.to_datetime('2011-06-06')), 'gender'] = 'F'

####
#The Clerk of the House:
# https://www.ourcommons.ca/About/Clerk/Clerk-History-e.htm

#Eric Janse, Charles Robert, Marc Bosc: 
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Clerk of the House') & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('2017-01-01')), 'gender'] = 'M'

#Audrey O'Brien
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Clerk of the House') & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('2006-01-01')) & 
             (pd.to_datetime(data_raw['date']) < pd.to_datetime('2017-01-01')), 'gender'] = 'F'

##### IS THE CHAIR THE SAME AS THE SPEAKER? THAT IS WHAT I AM ASSUMING FOR NOW

####
#The Chair:
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Chair'), 'gender'] = 'M'
    
####
#The Deputy Chair:
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Deputy Chair') & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('2012-09-17')), 'gender'] = 'M'
#Denise Savoie
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Deputy Chair') & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('2011-06-06')) & 
             (pd.to_datetime(data_raw['date']) < pd.to_datetime('2012-09-17')), 'gender'] = 'F'
#Andrew Scheer, William Blaikie, Charles Strahl
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Deputy Chair') & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('1994-01-18')) & 
             (pd.to_datetime(data_raw['date']) < pd.to_datetime('2011-06-06')), 'gender'] = 'F'
    
####
#The Assistant Deputy Chair/Assistant Deputy Speaker:
# Has been a man for a long time
#https://www.ourcommons.ca/procedure/procedure-and-practice-3/App05-e.html
    
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Assistant Deputy Chair'), 'gender'] = 'M'
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Assistant Deputy Speaker'), 'gender'] = 'M'

#Other non-MP speakers
data_raw.loc[data_raw['Current_Speaker'] == 'His Excellency Volodymyr Zelenskyy', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'H.E. Volodymyr Zelenskyy', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Raymonde Gagné', 'gender'] = 'F' #Speaker of the senate
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Joseph Biden, Jr.', 'gender'] = 'M' 
data_raw.loc[data_raw['Current_Speaker'] == 'Her Excellency Ursula von der Leyen', 'gender'] = 'F' 
data_raw.loc[data_raw['Current_Speaker'] == 'Right Hon. Justin Trudeau', 'gender'] = 'M' 
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Mona Fortier', 'gender'] = 'F' 
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Erin O\'Toole', 'gender'] = 'M' 
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Mark Holland', 'gender'] = 'M' 
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Michelle Rempel Garner', 'gender'] = 'F' 
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Ed Fast', 'gender'] = 'M' 
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Candice Bergen', 'gender'] = 'F'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Pierre Poilievre', 'gender'] = 'M'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Michelle Rempel Garner', 'gender'] = 'F'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Mary Ng', 'gender'] = 'F'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Chrystia Freeland', 'gender'] = 'F'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Anthony Rota', 'gender'] = 'M'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Bill Blair', 'gender'] = 'M'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Steven Guilbeault', 'gender'] = 'M'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Hedy Fry', 'gender'] = 'F'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Kerry-Lynne Findlay', 'gender'] = 'F'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Michael Chong', 'gender'] = 'M'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Marco Mendicino', 'gender'] = 'M'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Ahmed Hussen', 'gender'] = 'M'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Patty Hajdu', 'gender'] = 'F'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Sean Fraser', 'gender'] = 'M'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Carla Qualtrough', 'gender'] = 'F'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Karina Gould', 'gender'] = 'F'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Jonathan Wilkinson', 'gender'] = 'M'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Steven MacKinnon', 'gender'] = 'M'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Randy Boissonnault', 'gender'] = 'M'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Judy A. Sgro', 'gender'] = 'F'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Carolyn Bennett', 'gender'] = 'F'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Omar Alghabra', 'gender'] = 'M'  
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Pablo Rodriguez', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Marci Ien', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Kamal Khera', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. François-Philippe Champagne', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Joyce Murray', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Dan Vandal', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. David Lametti', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. John McKay', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Harjit S. Sajjan', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Ginette Petitpas Taylor', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Dominic LeBlanc', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Gudie Hutchings', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Jean-Yves Duclos', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Anita Anand', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Rob Moore', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Andrew Scheer', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Mike Lake', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Lisa Marie Barron', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Tim Uppal', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Marie-Claude Bibeau', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Lawrence MacAulay', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Marc Miller', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == "Hon. Seamus O'Regan", 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Filomena Tassi', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Andy Fillmore', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Mélanie Joly', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Diane Lebouthillier', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Kirsty Duncan', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Bardish Chagger', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Greg Fergus', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Robert Oliphant', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. David McGuinty', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Pascale St-Onge', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Jim Carr', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Marc Garneau', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Helena Jaczek', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. George J. Furey', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'John Brassard', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Arif Virani', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Soraya Martinez Ferrada', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Gary Anandasangaree', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Jenna Sudds', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Terry Beech', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Rechie Valdez', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == "Hon. Ya'ara Saks", 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Karina Gould (Leader of the Government in the House of Commons', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'Hon. Ruby Sahota', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'M. Xavier Barsalou-Duval', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'L’hon. Dan Vandal', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'M. Maxime Blanchette-Joncas', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Right Right Hon. Justin Trudeau', 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'] == 'Mme Louise Chabot', 'gender'] = 'F'
data_raw.loc[data_raw['Current_Speaker'] == 'M. Sébastien Lemire', 'gender'] = 'M'

data_raw.loc[data_raw['Speaker_Tag'] == 'The Presiding Officer (Mr. Louis Plamondon)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == "The Acting Speaker (Mr. Chris d'Entremont)", 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Speaker (Mrs. Carol Hughes)', 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Speaker (Mrs. Alexandra Mendès)', 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'] == 'Assistant Deputy Speaker (Mrs. Alexandra Mendès)', 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Speaker (Mr. Gabriel Ste-Marie)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Speaker (Mr. Jamie Schmale)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Speaker (Mr. Michael Barrett)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Speaker (Mr. Mike Morrice)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Speaker (Mr. John Nater)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Speaker (Ms. Alexandra Mendès)', 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Speaker (Mrs. Alexandra Mendès)', 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Presiding Officer (Hon. Louis Plamondon)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Speaker (Mr. Tony Baldinelli)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Speaker (Mrs. Karen Vecchio)', 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Speaker (Mr. Scott Reid)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Speaker (Mr. Rob Morrison)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Speaker (Mr. Tom Kmiec)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Speaker (Mr. Todd Doherty)', 'gender'] = 'M'


nogender = data_raw.loc[data_raw['gender'].isna()]
unique_speakers = nogender['Speaker_Tag'].unique()


#remove spaces at the beginning of utterances
data_raw['text_clean'] = data_raw['text_clean'].str.replace(r"^\s*", "", regex = True)

#remove blank observations
#verified that these are blank on the website
data_raw = data_raw.loc[data_raw['text_clean'].str.strip() != ""]
data_raw = data_raw.dropna(subset=['text_clean'])

############
# Responding to another MP?

# Check 1: Who was the last person who spoke? (Political party and gender)
data_raw = data_raw.sort_values(by=["date", "Block_ID", "Utterance_ID"])

block_speakers = data_raw.drop_duplicates(subset=["date", "Block_ID", 'gender', 'party']).copy()
block_speakers["last_Speaker"] = block_speakers.groupby("date")["Current_Speaker"].shift(1)
block_speakers["last_gender"] = block_speakers.groupby("date")["gender"].shift(1)
block_speakers["last_party"] = block_speakers.groupby("date")["party"].shift(1)
block_speakers = block_speakers[['date', 'Block_ID', 'Current_Speaker', 'last_Speaker', 'last_gender', 'last_party']]

data_raw = data_raw.merge(block_speakers[['date', 'Block_ID', 'last_Speaker', 'last_gender', 'last_party']], on=['date', 'Block_ID'], how="left")


# Check 2: See if they mentioned the previous speaker's party in any utterance
data_raw_filtered = data_raw[data_raw['last_party'] != 'none']

'''
#initialize a variable
data_raw['mention_last'] = np.nan

data_raw.loc[
    (data_raw["text_clean"].str.contains(r"\bLiberal\b|\bLiberals\b", case=False, regex=True)) & 
    (data_raw["last_party"] == "Lib."), "mention_last"] = 1

data_raw.loc[
    (data_raw["text_clean"].str.contains(r"\bConservative\b|\bConservatives\b", case=False, regex=True)) & 
    (data_raw["last_party"] == "CPC"), "mention_last"] = 1

data_raw.loc[
    (data_raw["text_clean"].str.contains(r"\bNDP\b|\bNew\sDemocrat\b", case=False, regex=True)) & 
    (data_raw["last_party"] == "NDP"), "mention_last"] = 1

data_raw.loc[
    (data_raw["text_clean"].str.contains(r"\bGreen\sParty\b", case=False, regex=True)) & 
    (data_raw["last_party"] == "GP"), "mention_last"] = 1

data_raw.loc[
    (data_raw["text_clean"].str.contains(r"\bBloc\sQuébécois\b", case=False, regex=True)) & 
    (data_raw["last_party"] == "BQ"), "mention_last"] = 1

#set the same value for mention_last for all utterances in the block
data_raw['mention_last'] = data_raw.groupby(['date','Block_ID'])['mention_last'].transform(lambda x: x.ffill())
data_raw['mention_last'] = data_raw.groupby(['date','Block_ID'])['mention_last'].transform(lambda x: x.bfill())

data_raw["mention_last"] = data_raw["mention_last"].fillna(0)

'''

# Party in power
data_raw['inpower'] = "Lib."
data_raw['inpower'] = np.where(data_raw['date'] < pd.Timestamp("2015-11-04"), "CPC", "Lib.")


data_raw['mention_Lib'] = (data_raw["text_clean"].str.contains(r"\bLiberal\b|\bLiberals\b", case=False, regex=True) &
                           (data_raw["party"] != "Lib.")).astype(int)

data_raw['mention_Con'] = (data_raw["text_clean"].str.contains(r"\bConservative\b|\bConservatives\b", case=False, regex=True) &
                            (data_raw["party"] != "CPC")).astype(int)

data_raw['mention_NDP'] = (data_raw["text_clean"].str.contains(r"\bNDP\b|\bNew\sDemocrat\b", case=False, regex=True) &
                           (data_raw["party"] != "NDP")).astype(int)

data_raw['mention_GP'] = (data_raw["text_clean"].str.contains(r"\bGreen\sParty\b", case=False, regex=True) &
                            (data_raw["party"] != "GP")).astype(int)

data_raw['mention_BQ'] = (data_raw["text_clean"].str.contains(r"\bBloc\b", case=False, regex=True) &
                            (data_raw["party"] != "BQ")).astype(int)

data_raw['mention_PM'] = ((data_raw["text_clean"].str.contains(r"\bTrudeau\b", case=False, regex=True) & (data_raw['inpower'] == 'Lib.') & (data_raw['party'] != data_raw['inpower'])) | 
                          (data_raw["text_clean"].str.contains(r"\bHarper\b", case=False, regex=True) & (data_raw['inpower'] == 'CPC') & (data_raw['party'] != data_raw['inpower'])) |
                          (data_raw["text_clean"].str.contains(r"\bPrime\sMinister\b", case=False, regex=True) & (data_raw['party'] != data_raw['inpower'])) |
                          (data_raw["text_clean"].str.contains(r"\bPrime\sMinister\b", case=False, regex=True) & (data_raw['party'] != data_raw['inpower']))
                          ).astype(int)
##Could add "government"

data_raw['mention_any'] = (data_raw[['mention_Lib', 'mention_Con', 'mention_NDP', 'mention_GP', 'mention_BQ', 'mention_PM']].any(axis=1).astype(int))







#######Identifying interruptions
data_raw["interrupted"] = data_raw["text_clean"].str.endswith("—")



#finally: collapse to block level
#length of utterance
#data_raw_block = data_raw.groupby("Block_ID")["text_clean"].agg(" ".join).reset_index()
#data_raw["utterance_length"] = data_raw["text_clean"].str.len()

Block_Level = data_raw.groupby(["date", "Block_ID"]).agg(
    text_clean=("text_clean", " ".join),  
    Current_Speaker=("Current_Speaker", "first"),  
    party=("party", "first"),
    gender=("gender", "first"),                    
    last_Speaker=("last_Speaker", "first"),
    last_gender=("last_gender", "first"),
    last_party=("last_party", "first"), 
    mention_Lib=("mention_Lib", "max"),
    mention_Con=("mention_Con", "max"),
    mention_NDP=("mention_NDP", "max"),
    mention_GP=("mention_GP", "max"),
    mention_BQ=("mention_BQ", "max"),
    mention_PM=("mention_PM", "max"),
    mention_any=("mention_any", "max"),
    interrupted=("interrupted", "max")
).reset_index()

#save to use for NLP steps
Block_Level.to_csv('text_forNLP.csv', index=False)

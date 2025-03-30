# Written by James Cabral and Chris Haun
# This program processes the data from the web scrape and prepares it for the NLP steps

#NOTE: Some of the implementation follows the cleaning steps from Bisbee et al. (2021)

import os 
import pandas as pd 
import numpy as np

os.chdir(r"C:\Users\jcabr\OneDrive - University of Toronto\Coursework\Winter\ECO2460\Empirical Project\Data")

'''
#merging webscrape files
webscrape1 = pd.read_csv("webscrape_data\session_40-1.csv") # November 2008 - March 2011
webscrape4 = pd.read_csv("webscrape_data\session_41-1.csv") # June 2011 - December 2024

merged_data = pd.concat([webscrape1, webscrape4], ignore_index=True)
merged_data["date"] = pd.to_datetime(merged_data["day"].astype(str) + " " +
                            merged_data["month"] + " " + 
                            merged_data["year"].astype(str), format="%d %B %Y")
merged_data = merged_data.sort_values(by="date")
merged_data.to_csv('web_scrape_full.csv', index=False)
'''


# import the raw data
data_raw = pd.read_csv("web_scrape_full.csv")

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

#initialize
data_raw['party'] = np.nan

#some of these are in French. Will fix them after
partytags = ['Lib.', 'BQ', 'CPC', 'PCC', 'NDP', 'NPD', 'PV', 'GP', 'Ind.'] 

#identify the parties in the speaker tags with the regex approach
# usual format is (name, surname, party)
party_pattern = r', (' + '|'.join(partytags) + r')'
data_raw['party'] = data_raw['Speaker_Tag'].str.extract(party_pattern, expand=False).fillna(data_raw['party'])

#fix the French ones
data_raw.loc[data_raw['party'] == 'NPD', 'party'] = 'NDP' 
data_raw.loc[data_raw['party'] == 'PCC', 'party'] = 'CPC' 
data_raw.loc[data_raw['party'] == 'PV', 'party'] = 'GP' 

#now, use the Current_Speaker variable (which doesn't have the brackets) to make sure that each speaker has the same party along the whole dataset
#ffill ensures that if someone switches party, code should not break
data_raw = data_raw.sort_values(by=['date', 'Block_ID', 'Utterance_ID'], axis=0)
data_raw['party'] = data_raw.groupby('Current_Speaker')['party'].transform(lambda x: x.ffill())
data_raw['party'] = data_raw.groupby('Current_Speaker')['party'].transform(lambda x: x.bfill())


#replace party with none if it's the speaker or if a special case
data_raw.loc[data_raw['Speaker_Tag'] == 'The Deputy Speaker', 'party'] = 'none'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Speaker', 'party'] = 'none'
data_raw.loc[data_raw['Speaker_Tag'] == 'Le Président', 'party'] = 'none'

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

noparty_speakers = [
    'H.E. Felipe Calderón Hinojosa (President of the United Mexican States)',
    'Hon. Noël A. Kinsella (Speaker of the Senate)',
    'Right Hon. David Cameron (Prime Minister of the United Kingdom of Great Britain and Northern Ireland)',
    'H.H. Aga Khan (49th Hereditary Imam of the Shia Imami Ismaili Muslims)',
    'H.E. Petro Poroshenko (President of Ukraine)',
    'H.E. François Hollande (President of the French Republic)',
    'Mr. Barack Obama (President of the United States of America)',
    'Hon. George Furey (Speaker of the Senate)',
    'The Assistant Deputy Speake (Mr. Anthony Rota )',
    'Le vice-président adjoint (M. Anthony Rota)',
    'Ms. Malala Yousafzai (Co-Founder of Malala Fund)',
    'The Assistant Deputy Speaker Mrs. (Carol Hughes)',
    'His Excellency Mark Rutte (Prime Minister of the Kingdom of the Netherlands)',
    'The Acting Chair (Mr. John Nater)',
    'The Acting Chair (Mrs. Carol Hughes)',
    'Assistant Deputy Speaker (Mrs. Alexandra Mendès)',
    'Assistant Deputy Speaker (Mrs. Carol Hughes)',
    'Hon. George J. Furey (Speaker of the Senate)',
    'Hon. Karina Gould (Leader of the Government in the House of Commons',
    'L’hon. Dan Vandal (for the Minister of Health)', 
    'The Assistant Deputy Chair of Committees of the Whole', 
    'The Acting Clerk of the House',
    'Deputy Speaker'
]

data_raw.loc[data_raw['Speaker_Tag'].isin(noparty_speakers), 'party'] = 'none'

#these all checked manually
data_raw.loc[data_raw['Speaker_Tag'] == 'M. André Bellavance', 'party'] = 'BQ'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Thomas Mulcair', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Michel Guimond', 'party'] = 'BQ'
data_raw.loc[data_raw['Speaker_Tag'] == 'Ms. Maria Mourani', 'party'] = 'BQ'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Marc Lemay', 'party'] = 'BQ'
data_raw.loc[data_raw['Speaker_Tag'] == 'Ms. Shelly Glover', 'party'] = 'CPC'
data_raw.loc[data_raw['Speaker_Tag'] == 'Ms Anne Minh-Thu Quach', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'Ms Laurin Liu', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'Ms. Anne-Marie Day', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'Ms Françoise Boivin', 'party'] = 'Lib.'
data_raw.loc[data_raw['Speaker_Tag'] == 'Ms. Sadia Groguhé', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'Ms. Carol Hughes', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'Ms. Djaouida Sellah', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Hoang Mai', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Pierre-Luc Dusseault', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == "L'hon. Lisa Raitt", 'party'] = 'CPC'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Garnett Genuis', 'party'] = 'CPC'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Kevin Lamoureux', 'party'] = 'Lib.'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Guy Caron', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Daniel Blaikie', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Tom Kmiec', 'party'] = 'CPC'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Robert Aubin', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'Peter Julian', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Joël Lightbound', 'party'] = 'Lib.'
data_raw.loc[data_raw['Speaker_Tag'] == 'Ms. Marilène Gill', 'party'] = 'BQ'
data_raw.loc[data_raw['Speaker_Tag'] == 'Jaime Battiste', 'party'] = 'Lib.'
data_raw.loc[data_raw['Speaker_Tag'] == 'Ted Falk', 'party'] = 'CPC'
data_raw.loc[data_raw['Speaker_Tag'] == 'M. Pierre Paul-Hus', 'party'] = 'CPC'
data_raw.loc[data_raw['Speaker_Tag'] == 'Lisa Marie Barron', 'party'] = 'NDP'
data_raw.loc[data_raw['Speaker_Tag'] == 'Ms. Tracy Gray', 'party'] = 'CPC'
data_raw.loc[data_raw['Speaker_Tag'] == 'John Brassard', 'party'] = 'CPC'
data_raw.loc[data_raw['Speaker_Tag'] == 'Mr. Alexis Brunelle‑Duceppe', 'party'] = 'BQ'

#check to see if there are still missing parties
noparty = data_raw.loc[data_raw['party'].isna()]
unique_speaker_tags = noparty['Speaker_Tag'].unique().tolist()

#####################
# Idntifying gender #
#####################
#for speakers, I am obtaining information from: 
#https://lop.parl.ca/sites/ParlInfo/default/en_CA/People/OfficersParliament/politicalOfficersCommons/deputySpeakers

#NOTE: Code will not work if this is being run for time periods beyond what we do in the paper if positions have changed

#initialize
data_raw['gender'] = np.nan

data_raw.loc[data_raw['Speaker_Tag'].str.startswith('Mr.'), 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'].str.startswith('Mr.'), 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'].str.startswith('M.'), 'gender'] = 'M'

data_raw.loc[data_raw['Speaker_Tag'].str.startswith('Mrs.'), 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'].str.startswith('Ms.'), 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'].str.startswith('Mme.'), 'gender'] = 'F'

####
#The Speaker: 
#No female speaker since Jeanne Sauve in 1980-1984
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Speaker'), 'gender'] = 'M'
data_raw.loc[(data_raw['Current_Speaker'] == 'The Acting Speaker'), 'gender'] = 'M'

####
#The Deputy Speaker
#Chris d'Entremon (still the DS), Bruce Stanton, Joseph Comartin
#https://lop.parl.ca/sites/ParlInfo/default/en_CA/People/OfficersParliament/politicalOfficersCommons/deputySpeakers
data_raw.loc[((data_raw['Speaker_Tag'] == 'The Deputy Speaker') | (data_raw['Speaker_Tag'] == 'Deputy Speaker')) & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('2012-09-17')), 'gender'] = 'M'
#Denise Savoie
data_raw.loc[((data_raw['Speaker_Tag'] == 'The Deputy Speaker') | (data_raw['Speaker_Tag'] == 'Deputy Speaker')) & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('2011-06-06')) & 
             (pd.to_datetime(data_raw['date']) < pd.to_datetime('2012-09-17')), 'gender'] = 'F'
#Andrew Scheer, William Blaikie, Charles Strahl
data_raw.loc[((data_raw['Speaker_Tag'] == 'The Deputy Speaker') | (data_raw['Speaker_Tag'] == 'Deputy Speaker')) & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('1994-01-18')) & 
             (pd.to_datetime(data_raw['date']) < pd.to_datetime('2011-06-06')), 'gender'] = 'F'

####
#The Clerk of the House:
# https://www.ourcommons.ca/About/Clerk/Clerk-History-e.htm

#Eric Janse, Charles Robert, Marc Bosc: 
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Clerk of the House') & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('2017-01-01')), 'gender'] = 'M'

data_raw.loc[(data_raw['Speaker_Tag'] == 'The Acting Clerk of the House') & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('2017-01-01')), 'gender'] = 'M'

#Audrey O'Brien
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Clerk of the House') & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('2006-01-01')) & 
             (pd.to_datetime(data_raw['date']) < pd.to_datetime('2017-01-01')), 'gender'] = 'F'

data_raw.loc[((data_raw['Speaker_Tag'] == 'The Acting Clerk of the House')) & 
             (pd.to_datetime(data_raw['date']) >= pd.to_datetime('2006-01-01')) & 
             (pd.to_datetime(data_raw['date']) < pd.to_datetime('2017-01-01')), 'gender'] = 'F'

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
data_raw.loc[(data_raw['Speaker_Tag'] == 'The Assistant Deputy Chair of Committees of the Whole'), 'gender'] = 'M'

#Other non-MP speakers
male_speakers = [
    'His Excellency Volodymyr Zelenskyy', 'H.E. Volodymyr Zelenskyy', 'Hon. Joseph Biden, Jr.',
    'Right Hon. Justin Trudeau', 'Hon. Erin O\'Toole', 'Hon. Mark Holland', 'Hon. Ed Fast',
    'Hon. Pierre Poilievre', 'Hon. Anthony Rota', 'Hon. Bill Blair', 'Hon. Steven Guilbeault',
    'Hon. Michael Chong', 'Hon. Marco Mendicino', 'Hon. Ahmed Hussen', 'Hon. Sean Fraser',
    'Hon. Jonathan Wilkinson', 'Hon. Steven MacKinnon', 'Hon. Randy Boissonnault',
    'Hon. Omar Alghabra', 'Hon. Pablo Rodriguez', 'Hon. François-Philippe Champagne',
    'Hon. Dan Vandal', 'Hon. David Lametti', 'Hon. John McKay', 'Hon. Harjit S. Sajjan',
    'Hon. Dominic LeBlanc', 'Hon. Jean-Yves Duclos', 'Hon. Rob Moore', 'Hon. Andrew Scheer',
    'Hon. Mike Lake', 'Hon. Tim Uppal', 'Hon. Lawrence MacAulay', 'Hon. Marc Miller',
    "Hon. Seamus O'Regan", 'Andy Fillmore', 'Hon. Greg Fergus', 'Hon. Robert Oliphant',
    'Hon. David McGuinty', 'Hon. Jim Carr', 'Hon. Marc Garneau', 'Hon. George J. Furey',
    'John Brassard', 'Hon. Arif Virani', 'Hon. Gary Anandasangaree', 'Hon. Terry Beech',
    'M. Xavier Barsalou-Duval', 'L’hon. Dan Vandal', 'M. Maxime Blanchette-Joncas',
    'Right Right Hon. Justin Trudeau', 'M. Sébastien Lemire',
    'Hon. Mauril Bélanger', 'Hon. Peter Milliken', 'Right Hon. Stephen Harper',
    'Hon. Stéphane Dion', 'Hon. Jack Layton', 'Hon. Ralph Goodale', 'Hon. Jay Hill',
    'Hon. Vic Toews', 'Hon. John McCallum', 'Hon. Christian Paradis', 'Hon. Gary Goodyear',
    'Hon. Geoff Regan', 'Hon. Larry Bagnell', 'Hon. Jim Flaherty',
    'Hon. Scott Brison', 'Hon. James Moore', 'Hon. Denis Lebel', 'Hon. Jim Prentice',
    'Hon. Steven Fletcher', 'Hon. Keith Martin', 'Hon. Shawn Murphy', 'Hon. Rob Nicholson',
    'Hon. Mark Eyking', 'Hon. John Baird', 'Hon. Irwin Cotler', 'Hon. Wayne Easter',
    'Hon. Tony Clement', 'Hon. Stockwell Day', 'Hon. Lawrence Cannon', "Hon. Gordon O'Connor",
    'Hon. Peter Kent', 'Hon. Joseph Volpe', 'Hon. Gurbax Malhi', 'Hon. Chuck Strahl',
    'Hon. Ujjal Dosanjh', 'Hon. Peter MacKay', 'Hon. Jason Kenney', 'Hon. Denis Coderre',
    'Hon. Dan McTeague', 'Hon. Peter Van Loan', 'Hon. Jim Karygiannis', 'Hon. Navdeep Bains',
    'Hon. Rob Merrifield', 'Hon. Gary Lunn', 'Hon. Jim Abbott', 'Hon. Ken Dryden',
    'Hon. Bob Rae', 'Hon. Gerry Ritz', 'Hon. Keith Ashfield', 'Hon. Bryon Wilfert', 'Hon. Maxime Bernier',
    'L\'hon. Denis Coderre', 'L\'hon. Lawrence Cannon', 'Hon. Michael Ignatieff', 'Hon. John Duncan',
    'Hon. Laurie Hawn', 'Hon. Rick Casson', 'Hon. Ted Menzies', 'Hon. Julian Fantino', 'Hon. Steven Blaney',
    'Hon. Joe Oliver', 'Hon. Bal Gosal', 'Hon. Greg Thompson', 'Right Hon. David Cameron', 'Hon. Peter Penashue',
    'Hon. Deepak Obhrai', 'Hon. Greg Rickford', 'Hon. Kevin Sorenson', 'Hon. Ed Holder', 'Hon. Khristinn Kellie Leitch',
    'Hon. K. Kellie Leitch', 'Hon. Bill Morneau', 'Hon. Thomas Mulcair', 'Hon. Ron Cannan', 'Hon. George Furey',
    'Hon. Andrew Leslie', 'Hon. Deb Schulte', 'Hon. Peter Julian', 'Ted Falk', 'Hon. Erin O’Toole',
    'Hon. Seamus O’Regan', 'Hon. Maurizio Bevilacqua', 'Hon. Noël A. Kinsella', 'Hon. Bernard Valcourt', 'Hon. Chris Alexander',
    'Hon. Denis Paradis', 'Hon. Amarjeet Sohi', 'Hon. Kent Hehr', 'Hon. Hunter Tootoo', 'Hon. Robert Nault',
    'The très hon. Justin Trudeau', 'L\'hon. Larry Bagnell', 'L\'hon. Denis Lebel', 'L\'hon. François-Philippe Champagne',
    'L\'hon. Navdeep Bains', 'His Excellency Mark Rutte', 'Peter Julian', 'Jaime Battiste',
    'H.E. François Hollande', 'H.E. Felipe Calderón Hinojosa', 'H.H. Aga Khan', 'H.E. Petro Poroshenko',
    'Le vice-président adjoint', 'Hon. Jean-Pierre Blackburn', 'Hon. Gerry Byrne' 'Le Président',
    'Le très hon. Justin Trudeau', 'L’hon. Bill Morneau', 'L’hon. François-Philippe Champagne']

female_speakers = [
    'Hon. Marlene Jennings', 'Hon. Diane Finley', 'Hon. Lisa Raitt', 'Hon. Leona Aglukkaq',
    'Hon. Anita Neville', 'Hon. Judy Sgro', 'Hon. Rona Ambrose', 'Hon. Helena Guergis',
    'Hon. Gail Shea', 'Hon. Maria Minna', 'Hon. Albina Guarnieri', 'Hon. Diane Ablonczy',
    'Hon. Judy Foote', 'Hon. Maryam Monsef', 'Hon. Jane Philpott', 'Hon. Jody Wilson-Raybould',
    'Hon. Patricia Hajdu', 'Hon. Catherine McKenna', 'Hon. MaryAnn Mihychuk', 'Hon. Shelly Glover',
    'Hon. Michelle Rempel', 'Hon. Kellie Leitch', 'Hon. Alice Wong', 'Mme Christiane Gagnon',
    'Ms Anne Minh-Thu Quach', 'Ms Laurin Liu', 'Ms Françoise Boivin', 'Mme Elizabeth May',
    'Hon. Kerry-Lynne D. Findlay', 'Mme Karen McCrimmon', 'Mme Eva Nassif', 'L’hon. Lisa Raitt',
    'L’hon. Diane Lebouthillier', 'L’hon. Bardish Chagger', 'Hon. Bernadette Jordan', 'Hon. Deb Schulte',
    'Mme Claude DeBellefeuille', 'Mme Yasmin Ratansi', 'Mme Anne Minh-Thu Quach',
     'L’hon. Rona Ambrose', 'Hon. Patricia Hajdu', 'L’hon. Bardish Chagger', 
    'L’hon. Diane Lebouthillier', 'L’hon. Lisa Raitt', 'Hon. Bernadette Jordan',
    'Hon. Josée Verner', 'Hon. Lynne Yelich', 'Hon. Bev Oda', 'Dr. Kellie Leitch', 'L\'hon. Lisa Raitt',
    'L\'hon. Rona Ambrose', 'Hon. Deb Schulte', 'Hon. Bernadette Jordan', 
    'Hon. Hedy Fry', 'Hon. Carolyn Bennett', 'Hon. Candice Bergen', 'The Acting Clerk of the House', 'Hon. Marie-Claude Bibeau',
    'Hon. Chrystia Freeland', 'Hon. Kirsty Duncan', 'Hon. Mélanie Joly', 'Hon. Bardish Chagger', 'Hon. Carla Qualtrough', 
    'Hon. Patty Hajdu', 'Hon. Diane Lebouthillier', 'Hon. Ginette Petitpas Taylor', 'Hon. Judy A. Sgro', 'Hon. Karina Gould',
    'Hon. Filomena Tassi', 'Hon. Mary Ng', 'Hon. Joyce Murray', 'Hon. Michelle Rempel Garner', 'Hon. Mona Fortier',
    'Hon. Anita Anand', 'Hon. Kerry-Lynne Findlay', 'Deputy Speaker', 'Hon. Marci Ien', 'Hon. Kamal Khera',
    'Hon. Gudie Hutchings', 'Lisa Marie Barron', 'Hon. Pascale St-Onge', 'Hon. Helena Jaczek', 'Her Excellency Ursula von der Leyen',
    'Hon. Soraya Martinez Ferrada', 'Hon. Jenna Sudds', 'Hon. Rechie Valdez', "Hon. Ya'ara Saks", 'Hon. Raymonde Gagné',
    'Hon. Karina Gould (Leader of the Government in the House of Commons', 'Hon. Ruby Sahota', 'Mme Louise Chabot']

data_raw.loc[data_raw['Current_Speaker'].isin(male_speakers), 'gender'] = 'M'
data_raw.loc[data_raw['Current_Speaker'].isin(female_speakers), 'gender'] = 'F'

#Manual checks
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

data_raw.loc[data_raw['Speaker_Tag'] == 'Hon. Gerry Byrne (Humber—St. Barbe—Baie Verte, Lib.)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'Hon. Gerry Byrne', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'Le Président', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Speaker (Mrs. Carol Hughes )', 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Speaker (Mr. Anthony Rota)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Speaker (Mr. Anthony Rota )', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Speaker (Mr. Anothony Rota)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Chair (Mr. Anthony Rota)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Speake (Mr. Anthony Rota )', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Speaker Mrs. (Carol Hughes)', 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Speaker(Mr. Anthony Rota)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Speaker (Mrs Carol Hughes)', 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Chair (Mr. John Nater)', 'gender'] = 'M'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Acting Chair (Mrs. Carol Hughes)', 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'] == 'The Assistant Deputy Chair (Mrs. Alexandra Mendès)', 'gender'] = 'F'
data_raw.loc[data_raw['Speaker_Tag'] == 'Assistant Deputy Speaker (Mrs. Carol Hughes)', 'gender'] = 'F'

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
# NOTE: Checked manually that this worked properly
# "Last speaker" does not carry over from one day to the next
data_raw = data_raw.sort_values(by=["date", "Block_ID", "Utterance_ID"])

block_speakers = data_raw.drop_duplicates(subset=["date", "Block_ID", 'gender', 'party']).copy()
block_speakers["last_Speaker"] = block_speakers.groupby("date")["Current_Speaker"].shift(1)
block_speakers["last_gender"] = block_speakers.groupby("date")["gender"].shift(1)
block_speakers["last_party"] = block_speakers.groupby("date")["party"].shift(1)
block_speakers = block_speakers[['date', 'Block_ID', 'Current_Speaker', 'last_Speaker', 'last_gender', 'last_party']]

data_raw = data_raw.merge(block_speakers[['date', 'Block_ID', 'last_Speaker', 'last_gender', 'last_party']], on=['date', 'Block_ID'], how="left")

# Party in power
data_raw['inpower'] = "Lib."
data_raw['inpower'] = np.where(data_raw['date'] < pd.Timestamp("2015-11-04"), "CPC", "Lib.")

#Dummy variables: Mentioned this party and not a member of that party
# Note that the non-MPs will be removed from the sample
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

data_raw['mention_any'] = (data_raw[['mention_Lib', 'mention_Con', 'mention_NDP', 'mention_GP', 'mention_BQ', 'mention_PM']].any(axis=1).astype(int))


#######Identifying interruptions
data_raw["interrupted"] = data_raw["text_clean"].str.endswith("—").astype(int)

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

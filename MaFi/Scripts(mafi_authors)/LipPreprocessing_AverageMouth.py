import pandas as pd

# Settings
path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/'
columns_use = ['Word','WordIPA_automatic', 'Response','ResponseIPA_automatic',
'Dist','Accuracy','Levenshtein','Phoneme_Correct', 'Phoneme_Correct_Percentage']
output = 'DataComplete_IPA_Dist_Average.csv'

def average_distance(data):
    averaged_data = data.groupby('Word').mean()
    averaged_data['MaFI'] = averaged_data['Dist']*-1
    averaged_data.to_csv(path+slice+output,header=True,mode='w')

# # Averaging each slice of corpus
#
# #slice = 'American/American_2022/'
# #slice = 'American/Anna/'
# #slice = 'British/British_2022/'
# #slice = 'British/Surprisal/'
# #slice = 'British/Replication/'
#
# data = pd.read_excel(path + slice + "DataComplete_IPA_Dist.xlsx", usecols=columns_use)
# average_distance(data)


# Averaging combined corpus

# ## American
# slice = 'American/'
#
# data_Anna = pd.read_excel(path + slice + "Anna/DataComplete_IPA_Dist.xlsx", usecols=columns_use)
# data_Anna['Source'] = 'A1'
#
# data_American2022 = pd.read_excel(path +  slice + "American_2022/DataComplete_IPA_Dist.xlsx",usecols=columns_use)
# data_American2022['Source'] = 'A2'
#
# data = pd.concat([data_Anna, data_American2022], axis=0)
# data.to_excel(path+ slice + 'DataComplete_IPA_Dist.xlsx')
# average_distance(data)


# ## British
# slice = 'British/'
#
# data_S1 = pd.read_excel(path + slice + "Replication/DataComplete_IPA_Dist.xlsx",usecols=columns_use)
# data_S1['Source'] = 'B1'
#
# data_S2 = pd.read_excel(path + slice + "Surprisal/DataComplete_IPA_Dist.xlsx",usecols=columns_use)
# data_S2['Source'] = 'B2'
#
# data_British2022 = pd.read_excel(path + slice + "British_2022/DataComplete_IPA_Dist.xlsx",usecols=columns_use)
# data_British2022['Source'] = 'B3'
#
# data = pd.concat([data_S1,data_S2, data_British2022], axis=0)
# data.to_excel(path+ slice + 'DataComplete_IPA_Dist.xlsx')
# average_distance(data)


# ## Complete: American + British
# slice = 'Combined/'
#
# data_S1['Variation'] = 'British'
# data_S2['Variation'] = 'British'
# data_British2022['Variation'] = 'British'
# data_Anna['Variation'] = 'American'
# data_American2022['Variation'] = 'American'
#
# data = pd.concat([data_S1,data_S2, data_British2022, data_Anna, data_American2022], axis=0)
# data.to_excel(path+slice+'DataComplete_IPA_Dist.xlsx')
# average_distance(data)


# ## 100 words
#
# # Alex & Grace corpus

# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/100_words/'
# slice = 'Alex_corpus/'
# data = pd.read_excel(path + slice +"DataComplete_IPA_Dist.xlsx")
# output = slice + 'DataComplete_IPA_Dist_Average.csv'

# # Alex & Grace new data
#
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/100_words/'
# slice = 'Grace_American/'
# data = pd.read_excel(path + slice +"DataComplete_IPA_Dist.xlsx")
# data.drop(columns = ['PartID'], inplace = True)
# output = slice + 'DataComplete_IPA_Dist_Average.csv'

# # 4 variations
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/100_words/'
# slice = 'Canadian/'
# data = pd.read_excel(path + slice +"DataComplete_IPA_Dist.xlsx")
# data.drop(columns = ['PartID'], inplace = True)
# output = slice + 'DataComplete_IPA_Dist_Average.csv'
#
# averageDist=data.groupby('Word').mean()
# averageDist.drop(columns = ['Unnamed: 0'], inplace = True)
# averageDist.to_csv(path+output,header=True,mode='w')
#
# output = slice + 'DataComplete_IPA_Dist_Average_individual.csv'
# averageDist=data.groupby(['Word', 'Speaker','Speaker_Gender']).mean()
# averageDist.drop(columns = ['Unnamed: 0'], inplace = True)
# averageDist.to_csv(path+output,header=True,mode='w')

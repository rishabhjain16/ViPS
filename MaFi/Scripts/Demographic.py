import pandas as pd

# # Settings for Surprisal mouth - Set 3
# path='/Users/claudia/Data/MouthRating_2020/Data_Surp/Demographics/'
# Lists = ['Demographics_1', 'Demographics_2', 'Demographics_3',
#          'Demographics_4', 'Demographics_5', 'Demographics_6']
# appendix = '.csv'
# output = 'Demographics_complete.csv'

# # Settings for Surprisal replication mouth - Set 2
# path='/Users/claudia/Data/MouthRating_2020/Data_SurpRep/Demographics/'
# Lists = ['Demographics_1', 'Demographics_2', 'Demographics_3',
#          'Demographics_4', 'Demographics_5', 'Demographics_6',
#          'Demographics_7', 'Demographics_8', 'Demographics_9,',
#          'Demographics_10', 'Demographics_11', 'Demographics_12',
#          'Demographics_13', 'Demographics_14', 'Demographics_15'
#          ]
# appendix = '.csv'
# output = 'Demographics_complete.csv'

# # Settings for 2022 Amercian data
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/American_2022/raw_data/'
# Lists = ['data_exp_82928-v7_questionnaire-6zlu']
# appendix = '.csv'
# output = 'Demographics_complete.csv'

# Settings for 2022 British data
path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/British_2022/raw_data/'
Lists = ['data_exp_82622-v7_questionnaire-p2qz']
appendix = '.csv'
output = 'Demographics_complete.csv'


data_parts=[]
for each in Lists:
    data_part=pd.read_csv(path+each+appendix,usecols=['Participant Private ID','Question Key','Response'])
    data_parts.append(data_part)
data=pd.concat(data_parts, ignore_index=True)

data.to_csv(path+output,mode='w')

# Exploring demographic data
data['Participant Private ID'].value_counts()

data[data['Question Key'] == 'Native'].groupby('Response').count()
data[data['Question Key'] == 'Gender'].groupby('Response').count()
data_age=data[data['Question Key'] == 'Age']
pd.to_numeric(data_age['Response']).mean()
pd.to_numeric(data_age['Response']).std()
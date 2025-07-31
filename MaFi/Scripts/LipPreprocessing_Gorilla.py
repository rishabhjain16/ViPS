import pandas as pd

# # This is for preprocessing of 2020 data
# lists = ['List1','List2','List3','List4','List5','List6','List7']
# path = '/Users/claudia/Data/MouthRating_2020/Data/'
# appendix = '_removeDate.xlsx'
# output = 'DataComplete.csv'

# # This is for preprocessing of 2022 American data
# lists = ['data_exp_82928-v7_task-q74v']
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/American_2022/raw_data/'
# appendix = '.csv'
# output = 'DataComplete.csv'

# # This is for preprocessing of 2022 British data
# lists = ['data_exp_82622-v7_task-1fey', 'data_exp_82622-v7_task-jid8','data_exp_82622-v7_task-whv7',
#          'data_exp_82622-v7_task-friz', 'data_exp_82622-v7_task-iga2']
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/British_2022/raw_data/'
# appendix = '.csv'
# output = 'DataComplete.csv'

# onlineData=pd.DataFrame()
# for each in lists:
#     onlineData_part=pd.read_csv(path+each+appendix,
#                                   usecols=['Response','ANSWER'])
#     onlineData_part=onlineData_part.dropna()
#     onlineData_part=onlineData_part[onlineData_part.Response!='VIDEO STARTED']
#     onlineData_part = onlineData_part[onlineData_part.Response != 'NO']
#     onlineData_part = onlineData_part[onlineData_part.Response != 'YES']
#
#     onlineData=onlineData.append(onlineData_part,ignore_index=True)
#
# onlineData.set_index('ANSWER',inplace=True)
# onlineData.sort_index(inplace=True)
# onlineData.to_csv(path+output,mode='w')

def part_preprocess (data):
    data_processed=data.dropna()
    data_processed=data_processed[data_processed.Response!='VIDEO STARTED']
    data_processed = data_processed[data_processed.Response != 'NO']
    data_processed = data_processed[data_processed.Response != 'YES']

    return data_processed
    

# # This is for preprocessing of 2022 100 words data, Alex
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/100_words/'
# output = 'DataComplete_Alex.xlsx'
#
# onlineData=pd.DataFrame()
#
# onlineData_part = pd.read_excel(path+'Alex/data_exp_94349-v3_task-jid8.xlsx',
#                                   usecols=['Response','ANSWER', 'Participant Private ID'])
# onlineData_part = part_preprocess(onlineData_part)
# onlineData_part['Speaker'] = 'Alex'
# onlineData_part['Speaker_Language'] = 'American'
# onlineData_part['Listener_Language'] = 'British'
#
# onlineData = onlineData_part
#
# onlineData.rename(columns={'ANSWER':'Word', 'Participant Private ID': 'PartID'}, inplace=True)
# onlineData.set_index('Word',inplace=True)
# onlineData.sort_index(inplace=True)
#
# onlineData.to_excel(path+output)

# This is for preprocessing of 2022 100 words data, Grace
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/100_words/'
# output = 'DataComplete_Grace.xlsx'
#
# onlineData=pd.DataFrame()
#
# onlineData_part = pd.read_excel(path+'Grace/data_exp_94350-v3_task-jid8.xlsx',
#                                   usecols=['Response','ANSWER', 'Participant Private ID'])
# onlineData_part = part_preprocess(onlineData_part)
# onlineData_part['Speaker'] = 'Grace'
# onlineData_part['Speaker_Language'] = 'British'
# onlineData_part['Listener_Language'] = 'American'
#
# onlineData = onlineData_part
#
# onlineData.rename(columns={'ANSWER':'Word', 'Participant Private ID': 'PartID'}, inplace=True)
# onlineData.set_index('Word',inplace=True)
# onlineData.sort_index(inplace=True)
#
# onlineData.to_excel(path+output)

# # This is for preprocessing of 2022 100 words data, American
# lists = ['American/list1/data_exp_92601-v3_task-jid8.xlsx', 'American/list2/data_exp_92602-v3_task-jid8.xlsx']
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/100_words/'
# language_variation = 'American'
# output = 'DataComplete_American.xlsx'
#
# onlineData=pd.DataFrame()
# for each in lists:
#     onlineData_part=pd.read_excel(path+each,
#                                   usecols=['Response','ANSWER','Video','Participant Private ID'])
#     onlineData_part = part_preprocess(onlineData_part)
#     onlineData = onlineData.append(onlineData_part, ignore_index=True)
#
# onlineData['Language_Variation'] = language_variation
# onlineData[['A', 'Speaker']] = onlineData['Video'].str.split('_', expand = True)
# onlineData['Speaker'] = onlineData['Speaker'].str.rstrip('.mp4')
# onlineData.drop(columns = ['A', 'Video'], inplace=True)
#
# onlineData['Speaker_Gender'] = onlineData['Speaker'].replace({'DavidM':'Male', 'Erica':'Female'})
# onlineData['Speaker'] = onlineData['Speaker'].replace({'DavidM':'American_1', 'Erica':'American_2'})
#
# onlineData.rename(columns={'ANSWER':'Word', 'Participant Private ID': 'PartID'}, inplace=True)
# onlineData.set_index('Word',inplace=True)
# onlineData.sort_index(inplace=True)
#
# onlineData.to_excel(path+output)

# # This is for preprocessing of 2022 100 words data, Australian
# lists = ['Australian/list1/data_exp_92451-v4_task-1fey.xlsx', 'Australian/list2/data_exp_90702-v7_task-jid8.xlsx']
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/100_words/'
# language_variation = 'Australian'
# output = 'DataComplete_Australian.xlsx'
#
# onlineData=pd.DataFrame()
# for each in lists:
#     onlineData_part=pd.read_excel(path+each,
#                                   usecols=['Response','ANSWER','Video','Participant Private ID'])
#     onlineData_part = part_preprocess(onlineData_part)
#     onlineData = onlineData.append(onlineData_part, ignore_index=True)
#
# onlineData['Language_Variation'] = language_variation
# onlineData[['A', 'Speaker']] = onlineData['Video'].str.split('_', expand = True)
# onlineData['Speaker'] = onlineData['Speaker'].str.rstrip('.mp4')
# onlineData.drop(columns = ['A', 'Video'], inplace=True)
#
# onlineData['Speaker_Gender'] = onlineData['Speaker'].replace({'Jared':'Male', 'Sabrena':'Female'})
# onlineData['Speaker'] = onlineData['Speaker'].replace({'Jared':'Australian_1', 'Sabrena':'Australian_2'})
#
# onlineData.rename(columns={'ANSWER':'Word', 'Participant Private ID': 'PartID'}, inplace=True)
# onlineData.set_index('Word',inplace=True)
# onlineData.sort_index(inplace=True)
#
# onlineData.to_excel(path+output)

# # This is for preprocessing of 2022 100 words data, British
# lists = ['British/list1/data_exp_92450-v4_task-1fey.xlsx', 'British/list2/data_exp_92170-v5_task-jid8.xlsx']
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/100_words/'
# language_variation = 'British'
# output = 'DataComplete_British.xlsx'
# 
# onlineData=pd.DataFrame()
# for each in lists:
#     onlineData_part=pd.read_excel(path+each,
#                                   usecols=['Response','ANSWER','Video','Participant Private ID'])
#     onlineData_part = part_preprocess(onlineData_part)
#     onlineData = onlineData.append(onlineData_part, ignore_index=True)
# 
# onlineData['Language_Variation'] = language_variation
# onlineData[['A', 'Speaker']] = onlineData['Video'].str.split('_', expand = True)
# onlineData['Speaker'] = onlineData['Speaker'].str.rstrip('.mp4')
# onlineData.drop(columns = ['A', 'Video'], inplace=True)
# 
# onlineData['Speaker_Gender'] = onlineData['Speaker'].replace({'Ed':'Male', 'Michi':'Female'})
# onlineData['Speaker'] = onlineData['Speaker'].replace({'Ed':'British_1', 'Michi':'British_2'})
# 
# onlineData.rename(columns={'ANSWER':'Word', 'Participant Private ID': 'PartID'}, inplace=True)
# onlineData.set_index('Word',inplace=True)
# onlineData.sort_index(inplace=True)
# 
# onlineData.to_excel(path+output)

# This is for preprocessing of 2022 100 words data, Canadian
lists = ['Canadian/list1/data_exp_93700-v2_task-jid8.xlsx', 'Canadian/list2/data_exp_93701-v2_task-jid8.xlsx']
path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/100_words/'
language_variation = 'Canadian'
output = 'DataComplete_Canadian.xlsx'

onlineData=pd.DataFrame()
for each in lists:
    onlineData_part=pd.read_excel(path+each,
                                  usecols=['Response','ANSWER','Video','Participant Private ID'])
    onlineData_part = part_preprocess(onlineData_part)
    onlineData = onlineData.append(onlineData_part, ignore_index=True)

onlineData['Language_Variation'] = language_variation
onlineData[['A', 'Speaker']] = onlineData['Video'].str.split('_', expand = True)
onlineData['Speaker'] = onlineData['Speaker'].str.rstrip('.mp4')
onlineData.drop(columns = ['A', 'Video'], inplace=True)

onlineData['Speaker_Gender'] = onlineData['Speaker'].replace({'Dave':'Male', 'Amanda':'Female'})
onlineData['Speaker'] = onlineData['Speaker'].replace({'Dave':'Canadian_1', 'Amanda':'Canadian_2'})

onlineData.rename(columns={'ANSWER':'Word', 'Participant Private ID': 'PartID'}, inplace=True)
onlineData.set_index('Word',inplace=True)
onlineData.sort_index(inplace=True)

onlineData.to_excel(path+output)


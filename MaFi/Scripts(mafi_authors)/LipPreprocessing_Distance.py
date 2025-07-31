import panphon
import pandas as pd
import panphon.distance
import epitran


# Settings path

# # For main corpus
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/American/American_2022/'
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/British/British_2022/'
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/British/Surprisal/'
path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/British/Replication/'
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/American/Anna/'

data=pd.read_excel(path + "DataComplete_IPA.xlsx",sheet_name="DataComplete")
output = 'DataComplete_IPA_Dist.xlsx'

# # # For 100 words analysis
#
# path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/100_words/'
#
# # ##  Grace & Alex corpus
# # slice = 'Alex_corpus/'
# # data=pd.read_excel(path + slice+ "Alex_corpus_checked.xlsx")
# # output = slice + 'DataComplete_IPA_Dist.xlsx'
#
# # ## Grace & Alex new data
# # slice = 'Grace_American/'
# # data=pd.read_excel(path + slice+ "DataComplete_Grace_checked.xlsx")
# # output = slice + 'DataComplete_IPA_Dist.xlsx'
#
# # ## 4 variations
# # slice = 'Canadian/'
# # data=pd.read_excel(path + slice+ "DataComplete_Canadian_checked.xlsx")
# # output = slice + 'DataComplete_IPA_Dist.xlsx'

# Add IPA per word
# Use epitran to translate English word into IPA
# Needs flite to translate English (only). See here fore more details https://pypi.org/project/epitran/
epi = epitran.Epitran('eng-Latn')
def get_IPA(x):
    if pd.isna(x) == False:
        IPA = epi.transliterate(x)
    else:
        IPA = 'NaN'
    return IPA
data['WordIPA_automatic']= data['Word'].apply(get_IPA)
data['ResponseIPA_automatic'] = data['Response'].apply(get_IPA)


# Calculating PanPhon distance
ft = panphon.FeatureTable()
dst = panphon.distance.Distance()

def panphon_dist(word, response):
    if pd.isna(response) == True:
        dist = 'NaN'
    else:
        dist = dst.jt_weighted_feature_edit_distance_div_maxlen(str(word), str(response))
    return dist

data['Dist'] = data.apply(lambda x: panphon_dist(x.WordIPA_automatic, x.ResponseIPA_automatic), axis = 1)

# # Exclusive for the IPA comparisons
# data['Dist_British'] = data.apply(lambda x: panphon_dist(x.WordIPA_British, x.ResponseIPA_British), axis = 1)
# data['Dist_American'] = data.apply(lambda x: panphon_dist(x.WordIPA_American, x.ResponseIPA_American), axis = 1)

# Calculating accuracy
data['Word'] = data['Word'].str.strip(' ')
data['Response'] = data['Response'].str.strip(' ')
data['Accuracy'] = data['Word'].str.lower() == data['Response'].str.lower()
data['Accuracy'] = data['Accuracy'].astype(int)

# Calculating correct phoneme
def phoneme_correct (x, y):
    phoneme_num = 0
    correct_phoneme = 0
    if pd.isna(x) == False and pd.isna(y) == False :
        for phon in x:
            if phon != 'ˈ' and phon != 'ˌ':
                phoneme_num = phoneme_num+1
                if phon in y:
                    correct_phoneme = correct_phoneme+1
        phoneme_correct_percentage = correct_phoneme/phoneme_num
    else:
        correct_phoneme = 'NaN'
        phoneme_correct_percentage = "NaN"
    return pd.Series([correct_phoneme, phoneme_correct_percentage])

data[['Phoneme_Correct', 'Phoneme_Correct_Percentage']] = data.apply(lambda x: phoneme_correct(x['WordIPA_automatic'], x['ResponseIPA_automatic']), axis=1)

# Calculating levinstein distance

import Levenshtein
def lev_dist(x,y):
    if pd.isna(x) == False and pd.isna(y) == False :
        dist = Levenshtein.distance(x,y)
    else:
        dist = "NaN"
    return dist

data['Levenshtein'] = data.apply(lambda x: lev_dist(x['WordIPA_automatic'], x['ResponseIPA_automatic']), axis=1)

data.to_excel(path+output)

import epitran
import pandas as pd

# Settings for complete data
path = '/Users/claudia/OneDrive - University College London/MouthStudy_2021/'
slice = 'American/'
data = pd.read_csv(path + slice+ 'DataComplete_IPA_Dist_Average.csv')
output = 'DataComplete_IPA_Dist_Average_phoneme.csv'

# Add control variables
control=pd.read_excel('/Users/claudia/OneDrive - University College London/MouthStudy_2021/controlVariables.xlsx', sheet_name='Data')
data=data.join(control.set_index('Word'), on='Word')

# Add IPA per word
# Use epitran to translate English word into IPA
# Needs flite to translate English (only). See here fore more details https://pypi.org/project/epitran/
epi = epitran.Epitran('eng-Latn')
data['IPA']=[epi.transliterate(each) for each in data['Word']]
phonemes=[]

for each in data['Word']:
    IPA=epi.word_to_tuples(each)
    phonemes_each=[]
    for i in range (0,len(IPA)):
        phoneme=IPA[i][3]
        if phoneme != '':
            phonemes_each.append(phoneme)
    phonemes.append(phonemes_each)

data['phonemes']=phonemes

# Add phonological features per word
frontList=['b','p','m','f','v']
roundList=['r','w','o','ɔ','u']

data['isFront'] = (data['phonemes']
                   .apply(lambda x: list(set(x).intersection(set(frontList))))
                   ).str.len() != 0
data['isRound'] = (data['phonemes']
                   .apply(lambda x: list(set(x).intersection(set(roundList))))
                   ).str.len() != 0

# Add viseme features (initial & rest) per word
lowerLipTuckList=['f','v']
protrusionList=['ʃ','t͡ʃ','d͡ʒ','w']
labialClosureList=['b','p','m']
mouthNarrowingList=['w']
lipRoundingList=['j','r','w']

data['lowerLipTuck'] = (data['phonemes']
                   .apply(lambda x: list(set(x).intersection(set(lowerLipTuckList))))
                   ).str.len() != 0
data['protrusion'] = (data['phonemes']
                   .apply(lambda x: list(set(x).intersection(set(protrusionList))))
                   ).str.len() != 0
data['labialClosure'] = (data['phonemes']
                   .apply(lambda x: list(set(x).intersection(set(labialClosureList))))
                   ).str.len() != 0
data['mouthNarrowing'] = (data['phonemes']
                   .apply(lambda x: list(set(x).intersection(set(mouthNarrowingList))))
                   ).str.len() != 0
data['lipRounding'] = (data['phonemes']
                   .apply(lambda x: list(set(x).intersection(set(lipRoundingList))))
                   ).str.len() != 0

lowerLipTuck=[]
protrusion=[]
labialClosure=[]
mouthNarrowing=[]
lipRounding=[]

for each in data['phonemes']:
    lowerLipTuck.append(int(each in lowerLipTuckList))
    protrusion.append(int(each in protrusionList))
    labialClosure.append(int(each in labialClosureList))
    mouthNarrowing.append(int(each in mouthNarrowingList))
    lipRounding.append(int(each in lipRoundingList))
data['lowerLipTuck']=lowerLipTuck
data['protrusion']=protrusion
data['labialClosure']=labialClosure
data['mouthNarrowing']=mouthNarrowing
data['lipRounding']=lipRounding


informativeFeatures_total=frontList+\
                    roundList+\
                    lowerLipTuckList+\
                    protrusionList+\
                    labialClosureList+\
                    mouthNarrowingList+\
                    lipRoundingList

informativeFeatures_phoneme=frontList+\
                    roundList

informativeFeatures_viseme=lowerLipTuckList+\
                    protrusionList+\
                    labialClosureList+\
                    mouthNarrowingList+\
                    lipRoundingList

data['infoCount_total'] = (data['phonemes']
                   .apply(lambda x: list(set(x).intersection(set(informativeFeatures_total))))
                   ).str.len()
data['infoPercent_total'] = data['infoCount_total']/data['phonemes'].str.len()

data['infoCount_phoneme']=(data['phonemes']
                   .apply(lambda x: list(set(x).intersection(set(informativeFeatures_phoneme))))
                   ).str.len()
data['infoPercent_phoneme']=data['infoCount_phoneme']/data['phonemes'].str.len()

data['infoCount_viseme']=(data['phonemes']
                   .apply(lambda x: list(set(x).intersection(set(informativeFeatures_viseme))))
                   ).str.len()
data['infoPercent_viseme']=data['infoCount_viseme']/data['phonemes'].str.len()

data.to_csv(path+slice+output,mode='w')

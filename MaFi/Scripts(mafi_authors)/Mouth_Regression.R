library('tidyverse')
library('readxl')
library('dplyr')
library('naniar')
library('stargazer')
library('emmeans')
library('forcats')
require(gridExtra)

####################################### Data Cleaning #######################################
# Reading & cleaning data
setwd('/Users/claudia/OneDrive - University College London/MouthStudy_2021/')
data_mouth = read_csv('Combined/DataComplete_IPA_Dist_Average_phoneme.csv')
data_mouth_Brit = read_csv('British/DataComplete_IPA_Dist_Average_phoneme.csv')
data_mouth_Amer = read_csv('American/DataComplete_IPA_Dist_Average_phoneme.csv')
duplicate = merge(data_mouth_Brit,data_mouth_Amer, by.x = "Word", by.y = "Word")

data_use = mutate(data_mouth,
                    # Continuous variables are scaled
                    Distance_z = -scale(Dist),
                    AoA_z =  Age_Of_Acquisition %>% na_if('#') %>% as.numeric() %>% scale(),
                    Frequency_z = scale(Log_Freq_HAL),
                    Ortho_Neighbour_z = scale(Ortho_N),
                    Phono_Neighbour_z = scale(Phono_N),
                    Phoneme_Num_z = scale(NPhon), 
                    infoPercent_total_z = scale(infoPercent_total),
                    infoPercent_phoneme_z = scale(infoPercent_phoneme),
                    infoPercent_viseme_z = scale(infoPercent_viseme),
                    # 
                    # # Categorical variables are dummy coded
                    isFront = as.factor(isFront),
                    isRound = as.factor(isRound),
                    # 
                    lowerLipTuck = as.factor(lowerLipTuck),
                    protrusion = as.factor(protrusion),
                    labialClosure = as.factor(labialClosure),
                    mouthNarrowing = as.factor(mouthNarrowing),
                    lipRounding = as.factor(lipRounding)

)


data_use = filter (data_use, AoA_z != 'NaN') # exclude 276 words without control variable value (esp. AOA)

####################################### Analysis 0: descriptive #######################################

data_mouth_check = read_excel('Combined/DataComplete_IPA_Dist.xlsx')
#data_mouth_check = read_csv('Combined/DataComplete_IPA_Dist_average.csv')

data_mouth_check_slice = filter(data_mouth_check, Source == 'B1'|Source =='B2'|Source =='B3')

data_mouth_check_slice_words = data_mouth_check_slice %>%
  filter(Dist!='NA') %>%
  group_by(Word) %>%
  summarise(mean_dist = mean(Dist),
            mean_accuracy = mean(Accuracy),
            mean_phoneme = mean(Phoneme_Correct_Percentage))

mean(data_mouth_check_slice_words$mean_dist)
sd(data_mouth_check_slice_words$mean_dist)
min(data_mouth_check_slice_words$mean_dist)
max(data_mouth_check_slice_words$mean_dist)

mean(data_mouth_check_slice_words$mean_accuracy)
sd(data_mouth_check_slice_words$mean_accuracy)

mean(data_mouth_check_slice_words$mean_phoneme)
sd(data_mouth_check_slice_words$mean_phoneme)

# Duplicates
cor.test(duplicate$Dist.x, duplicate$Dist.y) # significantly correlated, but r=0.47
cor.test(duplicate$Accuracy.x, duplicate$Accuracy.y) # significantly correlated, but r=0.46
cor.test(duplicate$Phoneme_Correct_Percentage.x, duplicate$Phoneme_Correct_Percentage.y) # significantly correlated, but r=0.52
cor.test(duplicate$Levenshtein.x, duplicate$Levenshtein.y) # significantly correlated, but r=0.57

# Correlation with control variables
library('corrplot')

data_mouth_cor = mutate(data_use,
                        Score = Distance_z,
                        AoA = AoA_z,
                        Freq = Frequency_z,
                        OrthN = Ortho_Neighbour_z,
                        PhonN = Phono_Neighbour_z,
                        NPhon = Phoneme_Num_z
)

data_mouth_cor = data_mouth_cor %>% select('Score', 'AoA', 'Freq', 'OrthN', 'PhonN', 'NPhon')

rquery.cormat(data_mouth_cor, type = 'full')
data_mouth_cor_M = cor.test(data_mouth_cor)
corrplot(data_mouth_cor_M, tl.col = 'black') 

####################################### Analysis 1: feature analysis #######################################


model1.1 = lm(Distance_z ~
                isRound + 
                isFront +
                Frequency_z + Phono_Neighbour_z + AoA_z + Phoneme_Num_z,
              data = data_use)
summary(model1.1)


# Analysis 1.2: effect of viseme
model1.2 = lm(Distance_z ~
                lowerLipTuck + 
                protrusion + 
                labialClosure + 
                #mouthNarrowing +
                lipRounding +
                Frequency_z + Phono_Neighbour_z + AoA_z + Phoneme_Num_z,
              data = data_use)

summary(model1.2)


kappa(model1.2)
car::vif(model1.2)

model1.3 = lm(Distance_z~
                infoPercent_phoneme_z +
                Frequency_z + Phono_Neighbour_z + AoA_z + Phoneme_Num_z,
              data=data_use)
summary(model1.3)

model1.4 = lm(Distance_z~
                infoPercent_viseme_z +
                Frequency_z + Phono_Neighbour_z + AoA_z + Phoneme_Num_z,
              data=data_use)
summary(model1.4)


# Plotting


preprocess_plot <- function(data){
  output = 
    data %>%
    mutate(
      Distance = Dist*-1,
      phon = case_when(
        isFront == TRUE ~ 'Front',
        isRound == TRUE ~ 'Round',
        isFront == FALSE & isRound == FALSE ~ 'Other'
      ),
      viseme = case_when(
        lipRounding == TRUE ~ "Lip Rounding",
        lowerLipTuck == TRUE ~ "Lower Lip Tuck",
        protrusion == TRUE ~ "Protrusion",
        labialClosure == TRUE ~ "Labial Closure",
        lowerLipTuck == FALSE & protrusion == FALSE & labialClosure == FALSE & lipRounding == FALSE ~ 'Other'
      )
    ) %>%
    mutate(
      phon = factor (phon, levels = c("Other", "Front", "Round")),
      viseme = factor (viseme, levels = c("Other", "Lower Lip Tuck", "Protrusion", "Labial Closure", "Lip Rounding"))
    )
}
data_mouth_plot = preprocess_plot(data_use)

p_phon =
data_mouth_plot %>%
  ggplot(aes (x = phon, y = Distance, fill = phon)) +
  labs(title = '',
       x = 'Phoneme Feature',
       y = 'MaFI Score',
       fill = 'Feature'
  ) +
  scale_fill_brewer(palette='Pastel1') +
  scale_x_discrete(breaks = NULL, limits = c('Other', 'Front', 'Round')) +
  geom_boxplot()

pdf('Plots/feature_phoneme.pdf', width = 4, height = 4) # Open a new pdf file
grid.arrange(p_phon, ncol = 1, nrow = 1) 
dev.off() 

# Double counting w (using mouth narrowing variable, which only have /w/)
data_mouth_plot_w = filter(data_mouth_plot, mouthNarrowing == '1')
data_mouth_plot_w$viseme = recode(data_mouth_plot_w$viseme, "Lip Rounding" = "Protrusion")
data_mouth_plot_vis = rbind(data_mouth_plot, data_mouth_plot_w)

p_vis =
  data_mouth_plot %>%
  ggplot(aes (x = viseme, y = Distance, fill = viseme)) +
  labs(title = '',
       x = 'Viseme Feature',
       y = 'MaFI Score',
       fill = 'Feature'
  ) +
  scale_fill_brewer(palette='Dark2') +
  scale_x_discrete(breaks = NULL, limits = c('Other', 'Protrusion', 'Labial Closure', 'Lip Rounding', 'Lower Lip Tuck')) +
  geom_boxplot()


pdf('Plots/feature_viseme.pdf', width = 6, height = 4) # Open a new pdf file
grid.arrange(p_vis, ncol = 1, nrow = 1) 
dev.off() 

p_phon_index =
  ggplot(data_mouth_plot,aes(infoPercent_phoneme,Distance))+
  #stat_summary(fun.data=mean_cl_normal) + 
  geom_smooth(method='lm', formula= y~x, color="#999999") +
  geom_point(color="#69b3a2",alpha=0.4,size=1) +
  scale_x_continuous(breaks = c(0,0.5,1), limits = c(0,1)) +
  scale_y_continuous(breaks=c(-1.5, -1, -0.5, 0)) +
  labs(title = '',
       x = 'Informativeness Load (Phoneme)',
       y = 'MaFI Score'
  ) +
  theme_bw()+ 
  theme(legend.position='none')

pdf('Plots/feature_phoneme_index.pdf', width = 4, height = 4) # Open a new pdf file
grid.arrange(p_phon_index, ncol = 1, nrow = 1) 
dev.off() 

p_vis_index =
  ggplot(data_mouth_plot,aes(infoPercent_viseme,Distance))+
  #stat_summary(fun.data=mean_cl_normal) + 
  geom_smooth(method='lm', formula= y~x, fullrange=TRUE, color="#999999") +
  geom_point(color="#EE9E64",alpha=0.4,size=1) +
  scale_x_continuous(breaks = c(0,0.5,1), limits = c(0,1)) +
  scale_y_continuous(breaks=c(-1.5, -1, -0.5, 0)) +
  labs(title = '',
       x = 'Informativeness Load (Viseme)',
       y = 'MaFI Score'
  ) +
  theme_bw()+ 
  theme(legend.position='none')

pdf('Plots/feature_viseme_index.pdf', width = 4, height = 4) # Open a new pdf file
grid.arrange(p_vis_index, ncol = 1, nrow = 1) 
dev.off() 

####################################### Analysis 2: corss listener & speaker analysis #######################################

# 100 words: IPA comparison from Alex & Grace corpus
data_100_Alex = read_csv('100_words/Alex_corpus/DataComplete_IPA_Dist_Average.csv')
data_100_Grace = read_csv('100_words/Grace_corpus/DataComplete_IPA_Dist_Average.csv')

cor.test(data_100_Alex$Dist, data_100_Alex$Dist_British) #0.96
cor.test(data_100_Alex$Dist, data_100_Alex$Dist_American) #0.96
cor.test(data_100_Alex$Dist_British, data_100_Alex$Dist_American) #0.98

cor.test(data_100_Grace$Dist, data_100_Grace$Dist_British) #0.97
cor.test(data_100_Grace$Dist, data_100_Grace$Dist_American) #0.97
cor.test(data_100_Grace$Dist_British, data_100_Grace$Dist_American) #0.98

# 100 words: Alex & Grace
data_100_Alex_British = read_csv('100_words/Alex_British/DataComplete_IPA_Dist_Average.csv')
data_100_Grace_American = read_csv('100_words/Grace_American/DataComplete_IPA_Dist_Average.csv')

data_100_Alex$Variation = 'American Speaker American Listener'
data_100_Grace$Variation = 'British Speaker British Listener'
data_100_Alex_British$Variation = 'American Speaker British Listener'
data_100_Grace_American$Variation = 'British Speaker American Listener'

data_100_cross_listener = bind_rows(
  data_100_Alex[,c(1,2,9)], 
  data_100_Grace[,c(1,2,9)],
  data_100_Alex_British[,c(1,2,7)],
  data_100_Grace_American[,c(1,2,7)]
)

data_100_cross_listener_wide = data_100_cross_listener %>%
  spread(Variation, Dist)

cor.test(data_100_cross_listener_wide$`American Speaker American Listener`, 
    data_100_cross_listener_wide$`American Speaker British Listener`) #0.83
cor.test(data_100_cross_listener_wide$`British Speaker American Listener`, 
    data_100_cross_listener_wide$`British Speaker British Listener`) #0.74

data_100_cross_listener$mean_dist = ave(data_100_cross_listener$Dist, data_100_cross_listener$Word)

plot_100_cross_listeners = 
  data_100_cross_listener%>%
  ggplot(aes(x = Variation, y = -Dist, color = fct_reorder(Word, mean_dist))) +
  geom_jitter(
    aes(color = fct_reorder(Word, mean_dist)), 
    alpha=0.4,size=3, 
    position = position_jitter(seed = 1)
  ) +
  xlab('Variation of English') +
  ylab('MaFI Score') +
  theme_bw()+
  theme(legend.position='none') +
  geom_text(
    #data=filter(data_100_ab_plot, word_order == "window"| word_order == 'key'),
    position = position_jitter(seed = 1),
    aes(label = fct_reorder(Word, mean_dist), hjust=-0.3, color=fct_reorder(Word, mean_dist)),
    size=2)

require(gridExtra)
pdf('/Users/claudia/OneDrive - University College London/MouthStudy_2021/100_words/Plots/cross_listener.pdf', width = 10, height = 7) # Open a new pdf file
grid.arrange(plot_100_cross_listeners, ncol = 1, nrow = 1) 
dev.off() 

# 100 words: 4 variations
data_100_British = read_csv('100_words/British/DataComplete_IPA_Dist_Average.csv')
data_100_American = read_csv('100_words/American/DataComplete_IPA_Dist_Average.csv')
data_100_Australian = read_csv('100_words/Australian/DataComplete_IPA_Dist_Average.csv')
data_100_Canadian = read_csv('100_words/Canadian/DataComplete_IPA_Dist_Average.csv')

data_100_British$Variation = 'British'
data_100_American$Variation = 'American'
data_100_Australian$Variation = 'Australian'
data_100_Canadian$Variation = 'Canadian'

data_100_cross_speaker = bind_rows(
  data_100_British[,c(1,2,7)], 
  data_100_American[,c(1,2,7)],
  data_100_Australian[,c(1,2,7)],
  data_100_Canadian[,c(1,2,7)]
)

data_100_cross_speaker_wide = data_100_cross_speaker %>%
  spread(Variation, Dist)

cor.test(data_100_cross_speaker_wide$British, data_100_cross_speaker_wide$American) #0.73
cor.test(data_100_cross_speaker_wide$British, data_100_cross_speaker_wide$Australian) #0.74
cor.test(data_100_cross_speaker_wide$British, data_100_cross_speaker_wide$Canadian) #0.71
cor.test(data_100_cross_speaker_wide$American, data_100_cross_speaker_wide$Australian) #0.69
cor.test(data_100_cross_speaker_wide$American, data_100_cross_speaker_wide$Canadian) #0.70
cor.test(data_100_cross_speaker_wide$Australian, data_100_cross_speaker_wide$Canadian) #0.67

data_100_cross_speaker$mean_dist = ave(data_100_cross_speaker$Dist, data_100_cross_speaker$Word)

plot_100_cross_speaker = 
  data_100_cross_speaker%>%
  ggplot(aes(x = Variation, y = -Dist, color = fct_reorder(Word, mean_dist))) +
  geom_jitter(
    aes(color = fct_reorder(Word, mean_dist)), 
    alpha=0.4,size=3, 
    position = position_jitter(seed = 1)
  ) +
  xlab('Variation of English') +
  ylab('MaFI Score') +
  theme_bw()+
  theme(legend.position='none') +
  geom_text(
    #data=filter(data_100_ab_plot, word_order == "window"| word_order == 'key'),
    position = position_jitter(seed = 1),
    aes(label = fct_reorder(Word, mean_dist), hjust=-0.3, color=fct_reorder(Word, mean_dist)),
    size=2)

require(gridExtra)
pdf('/Users/claudia/OneDrive - University College London/MouthStudy_2021/100_words/Plots/cross_speaker.pdf', width = 10, height = 7) # Open a new pdf file
grid.arrange(plot_100_cross_speaker, ncol = 1, nrow = 1) 
dev.off() 


# 100 words: 4 variations individuals
data_100_British_individual = read_csv('100_words/British/DataComplete_IPA_Dist_Average_individual.csv')
data_100_American_individual = read_csv('100_words/American/DataComplete_IPA_Dist_Average_individual.csv')
data_100_Australian_individual = read_csv('100_words/Australian/DataComplete_IPA_Dist_Average_individual.csv')
data_100_Canadian_individual = read_csv('100_words/Canadian/DataComplete_IPA_Dist_Average_individual.csv')

data_100_British_individual %>%
  arrange(Speaker) %>%
  summarise_each(funs(cor(.[Speaker == "British_1"], .[Speaker == "British_2"])), starts_with("Dist")) #0.63

data_100_American_individual %>%
  arrange(Speaker) %>%
  summarise_each(funs(cor(.[Speaker == "American_1"], .[Speaker == "American_2"])), starts_with("Dist")) #0.65

data_100_Australian_individual %>%
  arrange(Speaker) %>%
  summarise_each(funs(cor(.[Speaker == "Australian_1"], .[Speaker == "Australian_2"])), starts_with("Dist")) #0.49

data_100_Canadian_individual %>%
  arrange(Speaker) %>%
  summarise_each(funs(cor(.[Speaker == "Canadian_1"], .[Speaker == "Canadian_2"])), starts_with("Dist")) #0.60

# # =============== old script ====================
# 
# preprocess_plot <- function(data){
#   output = 
#     data %>%
#     mutate(
#       phon_ini = case_when(
#         isFront_ini == '1' ~ 'Front',
#         isRound_ini == '1' ~ 'Round',
#         isFront_ini == '0' & isRound_ini == '0' ~ 'Other'
#       ),
# 
#       phon_rest = case_when(
#         isFront_rest == '1' ~ 'Front',
#         isRound_rest == '1' ~ 'Round',
#         isFront_rest == '0' & isRound_rest == '0' ~ 'Other'
#       ),
#       # Double counting issue for viseme: /w/ appear in both lip rounding & protrusion, here include in lip rounding, double count later
#       viseme_ini = case_when(
#         lipRounding_ini == '1' ~ "Lip Rounding",
#         lowerLipTuck_ini == '1' ~ "Lower Lip Tuck",
#         protrusion_ini == '1' ~ "Protrusion",
#         labialClosure_ini == '1' ~ "Labial Closure",
#         lowerLipTuck_ini == '0' & protrusion_ini == '0' & labialClosure_ini == '0' & lipRounding_ini == '0' ~ 'Other'
#       ),
# 
#       viseme_rest = case_when(
#         lipRounding_rest == '1' ~ "Lip Rounding",
#         lowerLipTuck_rest == '1' ~ "Lower Lip Tuck",
#         protrusion_rest == '1' ~ "Protrusion",
#         labialClosure_rest == '1' ~ "Labial Closure",
#         lowerLipTuck_rest == '0' & protrusion_rest == '0' & labialClosure_rest == '0' & lipRounding_rest == '0' ~ 'Other'
#       )
#       )
#     ) %>%
#     mutate(
#       phon_ini = factor (phon_ini, levels = c("Other", "Front", "Round")),
#       phon_rest = factor (phon_rest, levels = c("Other", "Front", "Round")),
#       viseme_ini = factor (viseme_ini, levels = c("Other", "Lower Lip Tuck", "Protrusion", "Labial Closure", "Lip Rounding")),
#       viseme_rest = factor (viseme_rest, levels = c("Other", "Lower Lip Tuck", "Protrusion", "Labial Closure", "Lip Rounding"))
#     )
# }
# 
# mouth_plot <- function(unit, data){
#   if (unit == 'P'){
#     x_label = 'Phoneme Feature'
#     color_palete = 'Pastel1'
#     x_col = data$phon
#     fill_col = data$phon
#   } else {
#     x_label = 'Viseme Feature'
#     color_palete = 'Dark2'
#     x_col = data$viseme
#     fill_col = data$viseme
#   }
#   p = 
#     data %>%
#     ggplot(aes (x = x_col, y = Distance, fill = x_col)) +
#     labs(title = '',
#          x = x_label,
#          y = 'Mouth Informativeness Score',
#          fill = 'Feature'
#     ) +
#     scale_fill_brewer(palette=color_palete) +
#     scale_x_discrete(breaks = NULL) +
#     #scale_y_continuous(limits = c (0, 1.7)) +
#     geom_boxplot() + 
#     theme(legend.position='none')
# }
# 
# data_mouth_plot = preprocess_plot(data_use)
# 
# # Double counting w in initial position (using mouth narrowing variable, which only have /w/)
# data_mouth_plot_w_ini = filter(data_mouth_plot, mouthNarrowing_ini == '1')
# data_mouth_plot_w_ini$viseme_ini = recode(data_mouth_plot_w_ini$viseme_ini, "Lip Rounding" = "Protrusion")
# data_mouth_plot_vis_ini = rbind(data_mouth_plot, data_mouth_plot_w_ini)
# p_vis_ini = mouth_plot('V', 'ini', data_mouth_plot_vis_ini)
# 
# # Double counting w in rest position (using mouth narrowing variable, which only have /w/)
# data_mouth_plot_w_rest = filter(data_mouth_plot, mouthNarrowing_rest == '1')
# data_mouth_plot_w_rest$viseme_rest = recode(data_mouth_plot_w_rest$viseme_rest, "Lip Rounding" = "Protrusion")
# data_mouth_plot_vis_rest = rbind(data_mouth_plot, data_mouth_plot_w_rest)
# p_vis_rest = mouth_plot('V', 'rest', data_mouth_plot_vis_rest)
# 
# 
# setwd('/Users/claudia/OneDrive - University College London/MouthStudy_2021/Plots/Plots_NoLegend/A')
# require(gridExtra)
# 
# mouth_save <- function(item, item_name){
#   pdf(paste(item_name, '.pdf'), width = 4, height = 4) # Open a new pdf file
#   grid.arrange(item, ncol = 1, nrow = 1) 
#   dev.off() 
# }
# 
# mouth_save(p_phon_ini, 'phon_ini')
# mouth_save(p_phon_rest, 'phon_rest')
# mouth_save(p_vis_ini, 'vis_ini')
# mouth_save(p_vis_rest, 'vis_rest')
# mouth_save(p_phon_index, 'phon_load')
# mouth_save(p_vis_index, 'vis_load')
# 
# 
# setwd('/Users/claudia/OneDrive - University College London/MouthStudy_2021/Plots/Plots_NoLegend/')
# 
# p_phon_legend = mouth_plot('P', 'ini', data_mouth_plot) + theme(legend.position='bottom')
# pdf('phon_legend.pdf', width = 8, height = 8) # Open a new pdf file
# grid.arrange(p_phon_legend, ncol = 1, nrow = 1) 
# dev.off() 
# 
# p_vis_legend = mouth_plot('V', 'ini', data_mouth_plot) + theme(legend.position='bottom')
# pdf('vis_legend.pdf', width = 8, height = 8) # Open a new pdf file
# grid.arrange(p_vis_legend, ncol = 1, nrow = 1) 
# dev.off() 

# 
# setwd('/Users/claudia/OneDrive - University College London/MouthStudy_2021/Plots/plots_2022/american/')
# require(gridExtra)
# 
# mouth_save <- function(item, item_name){
#   pdf(paste(item_name, '.pdf'), width = 4, height = 4) # Open a new pdf file
#   grid.arrange(item, ncol = 1, nrow = 1) 
#   dev.off() 
# }
# 
# mouth_save(p_phon, 'phoneme')
# mouth_save(p_vis, 'viseme')
# mouth_save(p_phon_index, 'phon_load')
# mouth_save(p_vis_index, 'vis_load')
# 
# p_phon_legend = data_mouth_plot %>%
#   ggplot(aes (x = phon, y = Distance, fill = phon)) +
#   labs(title = '',
#        x = 'Phoneme Feature',
#        y = 'MaFI Score',
#        fill = 'Feature'
#   ) +
#   scale_fill_brewer(palette='Pastel1') +
#   scale_x_discrete(breaks = NULL, limits = c('Other', 'Round', 'Front')) +
#   geom_boxplot()+ 
#   theme(legend.position='bottom')
# 
# pdf('phon_legend.pdf', width = 8, height = 8) # Open a new pdf file
# grid.arrange(p_phon_legend, ncol = 1, nrow = 1) 
# dev.off() 
# 
# p_vis_legend = 
#   data_mouth_plot %>%
#   ggplot(aes (x = viseme, y = Distance, fill = viseme)) +
#   labs(title = '',
#        x = 'Viseme Feature',
#        y = 'MaFI Score',
#        fill = 'Feature'
#   ) +
#   scale_fill_brewer(palette='Dark2') +
#   scale_x_discrete(breaks = NULL, limits = c('Other', 'Protrusion', 'Labial Closure', 'Lip Rounding', 'Lower Lip Tuck')) +
#   geom_boxplot()+ 
#   theme(legend.position='bottom')
# 
# pdf('vis_legend.pdf', width = 8, height = 8) # Open a new pdf file
# grid.arrange(p_vis_legend, ncol = 1, nrow = 1) 
# dev.off() 




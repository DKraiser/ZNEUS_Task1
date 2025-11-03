# ZNEUS_Task1
This is a first task of zneus 2025 course (speeddating)

## Changelog
- ### 0.1 Data analysis 
Here are added some basic info prints about a dataset (shape, NA values...). 

- ### 0.2 The EDA was performed
The EDA has been carried out and some facts were found out:
Top correlations with 'match':
like                               0.305302
d_like                             0.280762
funny_partner                      0.270321
funny_o                            0.270256
attractive_o                       0.260927
attractive_partner                 0.260745
d_funny_o                          0.258448
d_funny_partner                    0.257854
guess_prob_liked                   0.254669
shared_interests_partner           0.251829
shared_interests_o                 0.251816
d_shared_interests_o               0.250300
d_shared_interests_partner         0.249905
d_guess_prob_liked                 0.245278
d_attractive_o                     0.240620
d_attractive_partner               0.240001
intelligence_o                     0.171304
intelligence_partner               0.171279
sinsere_o                          0.164673
sincere_partner                    0.164566
d_intelligence_o                   0.162962
d_intelligence_partner             0.162227
d_sinsere_o                        0.154958
d_sincere_partner                  0.154204
d_ambitous_o                       0.135341
d_ambition_partner                 0.134934
ambition_partner                   0.133883
ambitous_o                         0.133690
expected_num_matches               0.118074
met                                0.100608
d_expected_num_matches             0.089217
expected_num_interested_in_me      0.074084
d_expected_num_interested_in_me    0.060481
d_clubbing                         0.060142
clubbing                           0.055203
d_d_age                           -0.051487
intelligence                       0.051274
field_Law/Public Policy            0.051211
d_importance_same_race            -0.050102
importance_same_race              -0.048932
shared_interests_important        -0.047959
pref_o_shared_interests           -0.047953
funny_important                    0.041468
pref_o_funny                       0.041252
d_yoga                             0.038522
d_age                             -0.038239
d_art                              0.037896
yoga                               0.036324
d_intelligence                     0.036197
attractive                         0.035854



d_importance_same_race and importance_same_race are highly correlated: 0.94
d_importance_same_religion and importance_same_religion are highly correlated: 0.93
d_sports and sports are highly correlated: 0.92
d_art and art are highly correlated: 0.91
d_museums and museums are highly correlated: 0.91
d_hiking and hiking are highly correlated: 0.91
d_theater and theater are highly correlated: 0.90
d_exercise and exercise are highly correlated: 0.90
d_shopping and shopping are highly correlated: 0.90
d_concerts and concerts are highly correlated: 0.90
d_tvsports and tvsports are highly correlated: 0.90
d_reading and reading are highly correlated: 0.89
d_expected_happy_with_sd_people and expected_happy_with_sd_people are highly correlated: 0.89
d_yoga and yoga are highly correlated: 0.89
d_guess_prob_liked and guess_prob_liked are highly correlated: 0.89
d_clubbing and clubbing are highly correlated: 0.89
d_music and music are highly correlated: 0.89
d_dining and dining are highly correlated: 0.89
d_movies and movies are highly correlated: 0.88
d_ambition and ambition are highly correlated: 0.88
d_tv and tv are highly correlated: 0.88
d_attractive_partner and attractive_partner are highly correlated: 0.86
d_attractive_o and attractive_o are highly correlated: 0.86
d_expected_num_matches and expected_num_matches are highly correlated: 0.86
art and museums are highly correlated: 0.86
d_gaming and gaming are highly correlated: 0.85
d_expected_num_interested_in_me and expected_num_interested_in_me are highly correlated: 0.85
d_sincere and sincere are highly correlated: 0.84
d_intelligence and intelligence are highly correlated: 0.84
d_like and like are highly correlated: 0.84
d_attractive and attractive are highly correlated: 0.83
d_sincere_partner and sincere_partner are highly correlated: 0.83
d_sinsere_o and sinsere_o are highly correlated: 0.83
d_funny_partner and funny_partner are highly correlated: 0.82
d_funny_o and funny_o are highly correlated: 0.82
d_intelligence_partner and intelligence_partner are highly correlated: 0.81
d_intelligence_o and intelligence_o are highly correlated: 0.81
d_funny and funny are highly correlated: 0.81

Using this info we can remove many params from the data and use only the dataset:
Data columns (total 20 columns):

| Index | Column |
| --- | --- |
| 0 |  ambitous_o |                  
| 1 |  shared_interests_o |          
| 2 |  d_attractive_o |              
| 3 |  d_sinsere_o |                 
| 4 |  d_intelligence_o |            
| 5 |  d_funny_o |                   
| 6 |  d_ambitous_o |                
| 7 |  d_shared_interests_o |        
| 8 |  ambition_partner |            
| 9 |  shared_interests_partner |    
| 10 | d_attractive_partner |        
| 11 | d_sincere_partner |           
| 12 | d_intelligence_partner |      
| 13 | d_funny_partner |             
| 14 | d_ambition_partner |          
| 15 | d_shared_interests_partner |  
| 16 | d_like |                      
| 17 | d_guess_prob_liked |          
| 18 | met |                         
| 19 | match |        

- ### 0.3 The prototype of a NN
Now there is a prototype of the NN with a possibility of setting some parameters

- ### 0.4 Experiment tracking with wandb and first experiments
Now we can track the experiments and the results are:
1. Initial: with the following config is visible the NN memorizes the training part but not patterns - train loss is decreasing and val loss is growing - overfitting. Lets try to reduce the count of neurons
cfg={
  "batch_size": 16,
  "max_workers": 2,
  "dropout": 0.3,
  "seed": 42,
  "n_in": None,
  "n_hidden": 64,
  "n_out": 1,
  "learning_rate": 0.1,
  "epochs": 100
}

2. Reduced count of neurons in the hidden layer: the config with the count of 32 instead of 64 was used and the results are following: the decrease of training loss has slowed down and the increase of val loss slowed down too, but still grows. Another way will be tried - dropout + weight decay.

3. dropout = 0.2 and weight_decay = 0.0001. The overfitting has weakened.

4. The increasing of dropout to 0.3 is tried. Result is that increasing of dropout was not useful, so the next experiment is reducing the count of hidden neurons again. 

5. The count of hiddens of 16 was tried. The NN is almost not learning, so it is too small. In next experiments 32 hidden will be used

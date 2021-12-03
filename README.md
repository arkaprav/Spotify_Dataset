## Spotify_Dataset
In this datatset around 42000 songs are listed with their relevent details like artists, danceability, speechiness etc. We need to determine the popularity of new songs by the songs given. 
# pre-processing
there are some irrelevent features like track name, uri and date. We don't need them so we can simply cut them out. Then we need to check the correlation matrix for same features, although there aren't any same feature, then we need to scale the dataset into a ssingle scale for that MinMaxscaler is used here. After that we need to check the mutual info rgression to check the influence of the every feature upon our target feature (popularity). Here it seems all the features chosen have more or less influence on the target variable, so we can't ignore any feature here. 
# model selection
As this is a classifiation problem, we need to use some classification model. I have used-
1. Logistic Regression
2. Random Forest classifier
3. Support Vector Classifier
4. Neural Network
5. MLP Classifier
# Evaluation 
For evaluation of classification problem, there are two metrics log loss and confusion matrix are mostly used. Here we used Confusion matrix to evaluate our problem.
The scores are as follows
1. Logistic Regression -        [[3956, 2227],
                                [1145, 5002]]
2. Random Forest Classifier -   [[3956, 2227],
                                [1145, 5002]]
3. Support Vector Classifier -  [[4100, 2083],
                                [ 814, 5333]]
4. Neural Network-              [[4394, 1789],
                                [ 924, 5223]]
5. MLP Classifier-              [[4317, 1866],
                                [ 921, 5226]
From these results we can clearly say that Random Forest Classifier works very well here. 

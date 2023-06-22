# Spotify Regressor
## The XGBoost Regressor
The XGBoost Regressor in this project predicts the popularity rating of a top 50 Spotify song (rankings were collected in 2019) based on a variety of factors. The model has 5000 estimators, a learning rate of 0.001, and early stopping based on the test sets. Included in the **spotify_regressor.py** file is the model, along with graphs of its "deviation" from actual popularity ratings given both the training and test set. As used in this project, "deviation" simply refers to how far the regressor was from the actual popularity value; if the popularity value for a song was 91 and the model predicted a popularity value of 88, the deviation would equal three. A mean squared error assessment is also included in the file.

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/datasets/leonardopena/top50spotify2019. Credit for the dataset collection goes to **Deepak Depu**, **Arpita Gupta**, **Taner Sekmen**, and others on *Kaggle*. It includes multiple aspects of the top 50 songs on Spotify (in 2019), including:
- Artist's name
- Valence
- Liveliness
- Popularity
- Acousticness

It should be noted that the Pandas library couldn't read the data file in its original format due to special charactes included in some of the names of the songs. To account for this issue, I altered the dataset slightly and replaced special characters with non-special equivalents (this change didn't impact the model because song titles were not included in the training set). As a result, the CSV included in this repository is slightly different from the one that can be found at the link above. It should also be noted that all x-values (input values) were scaled with Scikit-Learn's **StandardScaler** before being fed to the regressor as part of data preprocessing.

## Libraries
These neural networks and XGBoost Regressor were created with the help of the Tensorflow, Scikit-Learn, and XGBoost libraries.
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
- XGBoost's Website: https://xgboost.readthedocs.io/en/stable/#
- XGBoost's Installation Instructions: https://xgboost.readthedocs.io/en/stable/install.html

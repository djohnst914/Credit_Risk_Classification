# Credit Risk Classification Report

## Purpose:
To use various techniques to train and evaluate models with imbalanced classes, ultimately leading to build a model that can identify the creditworthiness of borrowers.

## Libraries/Modules:
- Numpy
- Pandas
- Pathlib
  - Path
- Sklearn.metrics
  - balanced_accuracy_score
  - confusion_matrix
- Imblearn.metrics
  - classification_report_imbalanced
- Warnings

## Data:
A dataset of historical lending activity from a peer-to-peer lending services company was used for this purpose. I've predicted the creditworthiness of borrowers using a supervised machine learning `LogisticRegression` model. Two versions of the dataset were used, the first being the original data, and the second being resampled data using the `RandomOverSampler` module from the imbalanced-learn library.

## Variables:
For both cases, I got the count of the target classes, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report based off predictions made for both cases using the 'X_test' variable.

## Machine Learning Process Stages:
1. Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.
2. Check the balance of the labels variable (y) by using the value_counts function.
3. Split the data into training and testing datasets by using train_test_split.
4. Create a `LogisticRegression` Model with the Original Data

     1. Fit a logistic regression model by using the training data (X_train and y_train).
     2. Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.
     3. Evaluate the model’s performance by doing the following:
    - Calculate the accuracy score of the model.
    - Generate a confusion matrix.
    - Print the classification report.
5. Predict a `LogisticRegression` Model with Resampled Training Data
    
    1.  Use the `RandomOverSampler` module from the imbalanced-learn library to resample the data. 
    2. Use the `LogisticRegression` classifier and the resampled data to fit the model and make predictions.
    3. Evaluate the model’s performance by doing the following:
    - Calculate the accuracy score of the model.
    - Generate a confusion matrix.
    - Print the classification report.

## Results

* Machine Learning `LogisticRegression` Model with Original Data:
![Screenshot 2023-06-01 at 3 02 45 PM](https://github.com/djohnst914/github_upload/assets/123714457/e84c69a7-b8ed-4b06-89e9-9ba511f50494)

* Machine Learning `LogisticRegression` Model with Resampled Training Data:
![Screenshot 2023-06-01 at 3 02 56 PM](https://github.com/djohnst914/github_upload/assets/123714457/e75f6f72-8f47-451d-b6f7-292862697562)

## Summary

The machine learning model that seems to perform the best is the `LogisticRegression` Model with Resampled Training Data, and is the model I would reccomend using. All the metrics improved/increased from the original data. Key metrics like 'pre' (precision: the ratio of correctly predicted positive classes to the total number of positively predicted classes), and 'f1' (f1-score: the harmonic mean of precision and recall) all improved or stayed the same from the original dataset. The only metric that didn't improve is the 'rec' (recall: the ratio of correctly predicted positive classes to the total actual positive classes), but it is a very, very small decrease and in my opinion isn't an issue due to the improvement of all the other metrics from the original dataset to the resampled dataset. For more information on the metrics [click here.](https://dev.to/amananandrai/performance-measures-for-imbalanced-classes-2ojj)

With all this being said, it is important to take note of the fact that machine learning models can tend to have a bias toward the larger dataset class in an imbalanced data set, causing the model to excel at predicting the larger class of the imbalanced set (which we can see in both models). This could definitely be the reason both models did a better job at predicting the '0' class compared to the '1' class. So, is it more important to predict the `0`'s (larger class), or the `1`'s?(smaller class). My answer would be the `1`'s, due to the fact that we can more rely on the model to better predict the larger class. So going forward, we can put a little more emphasis on predicting the smaller classes if necessary by experimenting with different models/logistics/code and seeing what works best. 

## Contributor:
**Dylan Johnston**

[GitHub](https://github.com/djohnst914) 

Email: dylanhjjohnston@gmail.com

## License:

This project is licensed under the GNU General Public License v3.0. See [LICENSE](https://github.com/djohnst914/Credit_Risk_Classification/blob/main/LICENSE) file for details.
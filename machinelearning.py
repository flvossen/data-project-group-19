# Machine learning:
#Our goal is to develop a machine learning model within the context of the study 'Multi-omics of the gut microbial ecosystem in Inflammatory Bowel Diseases (IBD). The aim is to make a model that can predict the study group to which a patient belongs - UC (Ulcerative Colitis), CD (Crohn’s Disease), or nonIBD— based on relevant features extracted from the dataset (metadata). 

#Methodology:
#We used Python as the primary programming environment for this analysis. Two machine learning models were implemented and compared: Logistic Regression and Random Forest. Logistic Regression provides insights into the relationship between features and the target variable through interpretable coefficients, while Random Forest leverages an ensemble of decision trees to handle more complex patterns within the data. Model performance was evaluated using metrics such as accuracy, precision, recall, F1-score, and mean absolute error (MAE).

### Data exploration 
# import pandas as pd
import pandas as pd

# Load metadata
data = pd.read_csv('/kaggle/input/metadata-tsv/metadata.tsv',sep='\t')

# Display metadata
data.head()

# Check unique subjects
data.Subject

#### Making sure every subject has only one sample in the dataset 
# Problem: Some patients in this dataset have multiple samples. To ensure there is only one row per patient, we can combine the rows coming from one subject.

# First, we need to identify which patients (=subjects) have multiple samples:

# Number of unique Subject values
unique_subjects = metadata['Subject'].nunique()
print(f"Number of unique Subject values: {unique_subjects}")

# Frequency of each unique value
subject_counts = metadata['Subject'].value_counts()

# Subject values that occur multiple times
duplicates = subject_counts[subject_counts > 1]
print("Subjects that occur multiple times:")
print(duplicates)

# Now we want each subject to occur only once (i.e., 1 row per subject!).
# Of the numeric columns, calculate the average between the samples coming from the same patient.
# For categorical columns, we take the first value that occurs for the patient.

# Split numeric and non-numeric columns
numeric_cols = metadata.select_dtypes(include=['number']).columns
non_numeric_cols = metadata.select_dtypes(exclude=['number']).columns

# Group by 'Subject' and apply aggregations
aggregated_data = metadata.groupby('Subject').agg(
    {col: 'mean' for col in numeric_cols} | {col: 'first' for col in non_numeric_cols}
)

# Reset index only if 'Subject' is not already a column
if 'Subject' not in aggregated_data.columns:
    aggregated_data.reset_index(inplace=True)

# Display the aggregated data
print(aggregated_data.head())

# Check the shape of the aggregated data
aggregated_data.shape

# Filter all rows with Subject 'H4008' in the original dataset
original_rows = metadata[metadata['Subject'] == 'H4008']
print("Original dataset (metadata):")
print(original_rows)

# Filter the row with Subject 'H4008' in the aggregated dataset
aggregated_row = aggregated_data[aggregated_data['Subject'] == 'H4008']
print("\nAggregated dataset (aggregated_data):")
print(aggregated_row)

# Checking the number of rows for Subject H4008
# Number of rows in the original dataset
print(f"Number of rows in metadata for 'H4008': {len(original_rows)}")

# Number of rows in the aggregated dataset
print(f"Number of rows in aggregated_data for 'H4008': {len(aggregated_row)}")

# We see that in the aggregated dataset, the subject 'H4008' has only one row.

# Now change the name of the aggregated data to metadata_adjusted to avoid confusion:
metadata_adjusted = aggregated_data

# Looking at the shape of the dataset, how many rows and how many columns are there now?
metadata_adjusted.shape

print(metadata_adjusted)

# FEATURES 
# Firstly, we explored which features were interesting for predicting the studygroup (CD, UC, nonIBD). 

# For our machine learning model, the interesting features are: 
# * week_num
# * interval_days
# * Age at diagnosis
# * fecalcal
# * BMI_at_baseline
# * Height_at_baseline
# * Weight_at_baseline
# * Study.Group
# * Gender
# * Antibiotics
# * race
# * smoking status 

# Then we do a little bit of exploration on these features, do they have a lot of missing values?
# Check for the number of NA values in each column
na_counts = metadata_adjusted.isna().sum()

# Display the number of missing values for each column
print(na_counts)

# Display columns with NA values
na_counts = metadata_adjusted.isna().sum()
na_columns = na_counts[na_counts > 0]

# Print columns with missing values
print(na_columns)

# Unfortunately, many of the key features contain a substantial amount of missing values ("NA"). 
# If we choose to include these features in our predictive model, we would need to remove the rows with missing data. 
# However, this would significantly reduce the number of usable rows, so it may be more effective to focus on the features that don't have missing values.

# CONVERSION TO NUMERIC VALUES 
# We have to convert the non-numeric features to numeric features because: 
# * Logistic regression requires that the data fed into it consists of numerical values, as it performs mathematical operations on the features, such as calculating weights, coefficients, and other numerical measures.
# * Random forest can also not work with non-numeric data.

# We convert the non-numeric features that may be interesting for our machine learning model into numeric:
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Variables to encode
categorical_columns = ['Study.Group', 'Gender', 'Antibiotics', 'race', 'smoking status']

# Loop through the columns and apply label encoding
for column in categorical_columns:
    metadata_adjusted[f'{column}_encoded'] = label_encoder.fit_transform(metadata_adjusted[column])
    print(f"Mapping for {column}:")
    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    print()  # Empty line for readability

# View the result (e.g., the first few rows)
print(metadata_adjusted[[col for col in categorical_columns] + [f'{col}_encoded' for col in categorical_columns]].head())

# From now on, we need to use the new '-encoded' names for the columns:
# Study.Group -> 'Study.Group_encoded'
# Gender -> 'Gender_encoded'
# Antibiotics -> 'Antibiotics_encoded'
# race -> 'race_encoded'
# smoking status -> 'smoking status_encoded'

# MACHINE LEARNING MODEL 1 
# 1) Defining target value y and features X 
# For our first machine learning model:
# * our y = Study.Group_encoded
# * our IBD_features (X) = interval_days, visit_num, Gender_encoded, Antibiotics_encoded, race_encoded, BMI_at_baseline, Height_at_baseline, Weight_at_baseline

# Due to the missing values that are present in the features 'BMI_at_baseline, Height_at_baseline, Weight_at_baseline' we first need to remove the rows with missing values:
# Removing missing values in the features BMI_at_baseline, Height_at_baseline, Weight_at_baseline
metadata_adjusted_withoutNA = metadata_adjusted.dropna(subset=['BMI_at_baseline',
       'Height_at_baseline', 'Weight_at_baseline'], axis=0)

# Preparing the data for training a machine learning model by setting up the features (X) and the target variable (y)
y = metadata_adjusted_withoutNA['Study.Group_encoded']
IBD_features = ['interval_days',
       'visit_num', 'Gender_encoded',
       'Antibiotics_encoded', 'race_encoded', 'BMI_at_baseline',
       'Height_at_baseline', 'Weight_at_baseline']
X = metadata_adjusted_withoutNA[IBD_features]

# Get dimensions (number of rows, number of columns) of y
y.shape

# Get dimensions (number of rows, number of columns) of X
X.shape

# With this we can see that after removing the missing values, we are left with 81 remaining rows. And we can see that we have 8 features that are represented by the columns.

# 2) Splitting data into train and test sets 
# Then we split our dataset into training and test sets for building our machine learning model:
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# 3) PREPROCESSING DATA: Normalize feature values 
# Next, we take a look at the distribution of our feature values, both for the train set and the test set
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Plotting the distribution of the train set 
plt.figure(figsize=(18, 6))  # Set the size of the plot
train_X.sample(8, axis="columns").boxplot()
plt.xticks(rotation=90)  # Rotate the gene names on the x-axis
plt.show()  # Show the plot

# Plotting the distribution of the test set 
plt.figure(figsize=(18, 6))  # Set the size of the plot
val_X.sample(8, axis="columns").boxplot()
plt.xticks(rotation=90)  # Rotate the gene names on the x-axis
plt.show()  # Show the plot

# The features clearly have very different scales. It is good practice to normalize the range of the features in the dataset.
# We can do this with the Scikit-learn object StandardScaler:
from sklearn.preprocessing import StandardScaler

# First we create a StandardScaler object called scaler_std:
scaler_std = StandardScaler()

# We first compute the mean and standard deviation from the train set:
scaler_std.fit(train_X)

print("Means:")
print(scaler_std.mean_)
print("Variances:")
print(scaler_std.var_)

# We then compute the mean and standard deviation from the test set:
scaler_std.fit(val_X)

print("Means:")
print(scaler_std.mean_)
print("Variances:")
print(scaler_std.var_)

# Next, we normalize the features in the train and test set.
# The scaler object has a function transform() that uses the means and variances computed by the fit() function to normalize the features and we can make a Pandas DataFrame with the original column names as follows:
# Normalizing the features of the train set 
train_X_std = scaler_std.transform(train_X)
train_X_std = pd.DataFrame(train_X_std, columns=train_X.columns)

train_X_std

# Normalizing the features of the test set 
val_X_std = scaler_std.transform(val_X)
val_X_std = pd.DataFrame(val_X_std, columns=train_X.columns)

val_X_std

# We can now plot the normalized features:
# Plotting distribution of the features of the train set 
plt.figure(figsize=(18, 6))
train_X_std.sample(8, axis="columns").boxplot()
plt.xticks(rotation=90)
plt.show()

# Plotting distribution of the features of the test set
plt.figure(figsize=(18, 6))
val_X_std.sample(8, axis="columns").boxplot()
plt.xticks(rotation=90)
plt.show()

# As we can see, the feature value distributions are now normalized for both the train set and the test set.

# 4) FIT LOGISTIC REGRESSION MODEL
# Import the LogisticRegression model
from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
cls_std = LogisticRegression()

# Fit the model on the normalized train set (features train_X_std and target train_y)
cls_std.fit(train_X_std, train_y)

# Optionally, print the model's parameters to check if it trained successfully
print("Model coefficients: ", cls_std.coef_)
print("Model intercept: ", cls_std.intercept_)

# Visualize model coefficients and intercepts in plots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Feature names and model parameters
feature_names = train_X_std.columns  # Assuming train_X_std is a DataFrame
coefficients = cls_std.coef_
intercepts = cls_std.intercept_

# Convert coefficients to a DataFrame for better visualization
coeff_df = pd.DataFrame(coefficients, columns=feature_names)
coeff_df['Class'] = [f"Class {i}" for i in range(coefficients.shape[0])]

# Melt the DataFrame for easier plotting
coeff_melted = coeff_df.melt(id_vars='Class', var_name='Feature', value_name='Coefficient')

# Plot coefficients
plt.figure(figsize=(12, 8))
sns.barplot(data=coeff_melted, x='Feature', y='Coefficient', hue='Class')
plt.title("Logistic Regression Coefficients per Class")
plt.xticks(rotation=45, ha="right")
plt.axhline(0, color='grey', linestyle='--')  # Line at 0 for reference
plt.tight_layout()
plt.show()

# Display intercepts separately
plt.figure(figsize=(6, 4))
sns.barplot(x=[f"Class {i}" for i in range(len(intercepts))], y=intercepts, palette="muted")
plt.title("Logistic Regression Intercepts per Class")
plt.ylabel("Intercept Value")
plt.axhline(0, color='grey', linestyle='--')
plt.tight_layout()
plt.show()

#### INTERPRETATION MODEL PARAMETERS
#The model parameters represent a multiclass logistic regression with three classes:

#Model Coefficients:
#* Each row corresponds to one class, and each column represents a feature.
#* Positive coefficients: Features that increase the probability of predicting a given class as the feature value increases. -> Example: "Gender_encoded" positively influences Class 0.
#* Negative coefficients: Features that decrease the probability of predicting a given class as the feature value increases. -> Example: "Gender_encoded" has a strong negative influence for Class 1.
#* For example, the feature with a positive coefficient (e.g., 0.291 for class 1) increases the odds of predicting class 1, while a negative coefficient (e.g., -0.547 for class 2) reduces it.
#* Larger bars (either positive or negative) indicate features that have a stronger influence on the prediction for that class. Smaller bars (near zero) indicate that the feature has little impact on the model's prediction for that class. -> "Height_at_baseline" has a large positive coefficient for Class 1, meaning it strongly increases the likelihood of predicting Class 1. -> "BMI_at_baseline" has a large positive coefficient for Class 2.
#Model Intercepts:
#* The intercepts represent the baseline log-odds for each class when all features are zero.
#* The intercept for class 1 (0.789) means this class has a higher likelihood of being predicted by default compared to classes 2 and 3 with lower intercepts (-0.521, -0.268).

#The model appears to have learned meaningful relationships, as indicated by the variation in coefficients. However, some features dominate the predictions, which may increase the risk of overfitting or reliance on a small set of features.

# Computing the accuracy:
# Compute predictions on the validation set val_X_std
predictions_std = cls_std.predict(val_X_std)

# Optionally, print the predicted class labels for each row in val_X_std
print(predictions_std)

# Alternatively, to get the class probabilities instead of predicted classes:
predictions_std_proba = cls_std.predict_proba(val_X_std)

# Display the first 10 predicted class probabilities (for both class 0 and class 1)
print(predictions_std_proba[:10])

# Compute the accuracy of the model's predictions on the validation set
accuracy = cls_std.score(val_X_std, val_y)
print("Accuracy: {}".format(accuracy))

#### INTERPRETATION ACCURACY
# The accuracy of this machine learning model is 42.86% (approximately 0.43), meaning the model correctly classified about 43% of the data points
# This low accuracy suggests that the model performs only slightly better than random guessing for a multi-class classification problem
# This also suggests that the model is struggling to generalize well, particularly in distinguishing between classes.

# We calculate some other metrics to evaluate the model training
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

precision = precision_score(val_y, predictions_std, average='weighted')
recall = recall_score(val_y, predictions_std, average='weighted')
f1 = f1_score(val_y, predictions_std, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Confusion Matrix
cm = confusion_matrix(val_y, predictions_std)
print("Confusion Matrix:\n", cm)

#### INTERPRETATION OTHER METRICS
# Precision = 0.561 => When the model predicts a class as positive, it is correct 56.1% of the time. This indicates moderate precision.
# Recall = 0.429 => The model correctly identifies 42.9% of the actual positive cases, showing it misses many true positives.
# F1-Score = 0.330 => The harmonic mean of precision and recall is low (33%), indicating overall poor balance between precision and recall.
# The confusion matrix shows:
# Class 0 is predicted reasonably well (8 correct, 1 misclassified).
# Class 1 is poorly predicted (7 misclassified as Class 0, only 1 correct).
# Class 2 is completely misclassified, with all 4 instances predicted as Class 0.
# Overall, the model struggles to distinguish between classes, particularly Classes 1 and 2.

### 5) FIT RANDOM FOREST MODEL
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

IBD_model = RandomForestRegressor(random_state=1)
IBD_model.fit(train_X_std, train_y)
IBD_preds = IBD_model.predict(val_X_std)
print(mean_absolute_error(val_y, IBD_preds))

#### INTERPRETATION MAE (Mean Absolute Error)
# The Mean Absolute Error (MAE) = 0.667
# This indicates that, on average, the predicted class labels deviate by 0.67 units from the true class labels (where 0 = CD, 1 = UC, and 2 = nonIBD)
# This suggests that the model's predictions are not highly accurate, and there is room for improvement in classifying the conditions correctly
# Lower MAE values would indicate better model performance

## MACHINE LEARNING MODEL 2
# Our first model did not perform well. For our next model, we will try to improve by using fewer features by excluding 'BMI_at_baseline', 'Height_at_baseline', and 'Weight_at_baseline'.
# This allows us to retain more data, as we no longer need to remove rows with missing values for these features.

### 1) Defining target value y and features X
# For our first machine learning model:
# Our target value y = Study.Group_encoded
# Our IBD_features (X) = interval_days, visit_num, Gender_encoded, Anitbiotics_encoded, Race_encoded

# Preparing the data for training a machine learning model by setting up the features (X) and the target variable (y)
y = metadata_adjusted['Study.Group_encoded']
IBD_features = ['interval_days',
                'visit_num', 'Gender_encoded',
                'Antibiotics_encoded', 'race_encoded']
X = metadata_adjusted[IBD_features]

#get dimensions (number of rows, number of columns) of y
y.shape

#get dimensions (number of rows, number of columns) of X
X.shape

# We see that we have 105 rows and 5 features that are represented as the columns

### 2) Splitting data into train and test sets
# Then we split our dataset into training and test sets for building our machine learning model:
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# ### 3) PREPROCESSING DATA: Normalize feature values 
# Next, we take a look at the distribution of our feature values, both for the train set and the test set

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# # Plotting the distribution of the train set 
plt.figure(figsize=(18,6))  # Set the size of the plot
train_X.sample(5, axis="columns").boxplot()
plt.xticks(rotation=90)  # Rotate the gene names on the x-axis
plt.show()  # Show the plot

# # Plotting the distribution of the test set 
plt.figure(figsize=(18,6))  # Set the size of the plot
val_X.sample(5, axis="columns").boxplot()
plt.xticks(rotation=90)  # Rotate the gene names on the x-axis
plt.show()  # Show the plot

# The features clearly have very different scales. It is good practice to normalize the range of the features in the dataset.

# Importing the Scikit-learn object StandardScaler
from sklearn.preprocessing import StandardScaler

# # First we create a StandardScaler object called scaler_std
scaler_std = StandardScaler()

# # We first compute the mean and standard deviation from the train set
scaler_std.fit(train_X)

print("Means:")
print(scaler_std.mean_)
print("Variances:")
print(scaler_std.var_)

# # We then compute the mean and standard deviation from the test set
scaler_std.fit(val_X)

print("Means:")
print(scaler_std.mean_)
print("Variances:")
print(scaler_std.var_)

# Next, we normalize the features in the train and test set.

# # Normalizing the features of the train set 
train_X_std = scaler_std.transform(train_X)
train_X_std = pd.DataFrame(train_X_std, columns=train_X.columns)

train_X_std

# # Normalizing the features of the test set 
val_X_std = scaler_std.transform(val_X)
val_X_std = pd.DataFrame(val_X_std, columns=train_X.columns)

val_X_std

# # Plotting distribution of the features of the train set 
plt.figure(figsize=(18,6))
train_X_std.sample(5, axis="columns").boxplot()
plt.xticks(rotation=90)
plt.show()

# # Plotting distribution of the features of the test set
plt.figure(figsize=(18,6))
val_X_std.sample(5, axis="columns").boxplot()
plt.xticks(rotation=90)
plt.show()

# As we can see, the feature value distributions are now normalized for both the train set and the test set.



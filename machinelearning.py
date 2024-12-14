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

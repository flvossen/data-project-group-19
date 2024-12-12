# data-project-group-19

# Research question 1 (univariate analysis): Can calprotectin levels be used as a diagnostic biomarker to differentiate between non-IBD, ulcerative colitis (UC), and Crohn's disease (CD) in patients?

**Null hypothesis:** There is no significant difference in fecal calprotectin levels between patients with CD (Crohn's Disease), UC (Ulcerative Colitis), and non-IBD (individuals without inflammatory bowel diseases).

**Alternative hypothesis:** There is a significant difference in fecal calprotectin levels between patients with CD (Crohn's Disease), UC (Ulcerative Colitis), and non-IBD (individuals without inflammatory bowel diseases).

## Data preprocessing:

Read data:
```{r}
metadata <- read_tsv('metadata.tsv')
```

Do we have missing values, and if so, in which columns?
```{r}
sum(is.na(metadata))
colnames(metadata)[colSums(is.na(metadata))>0]
```
We have 749 missing values in the next columns: 'consent_age','Age at diagnosis', 'fecalcal', 'BMI_at_baseline', Height_at_baseline', 'Weight_at_baseline' and 'smoking status'.

Amount of missing values for each column:
```{r}
col_na_count <- colSums(is.na(metadata))
barplot(col_na_count, main = "Amount of missing values for each column", col = "lightblue", names.arg = colnames(metadata), las = 2, cex.names = 0.5)
```

_hoe hebben we deze NA's gezien en waarom enkel in fecalcal verwijderen?._
Fecal calprotectin contains NA values, we will remove them.
```{r}
clean_data1 <- metadata %>% filter(!is.na(fecalcal)) 
print(clean_data1) 
```
















# Research question 2 (multivariate analysis): Are bacterial classes associated with Non-IBD, Ulcerative Colitis (UC), and Crohn's Disease (CD) in terms of microbiome composition?

**Null hypothesis:** There is no significant difference in bacterial composition between the three study groups (class CD, UC and non-IBD).

**Alternative hypothesis:** There is significant difference in bacterial composition between the three study groups (class CD, UC and non-IBD).

## Data preprocessing:

Read data
```{r}
genera_counts <- read_tsv('genera.counts.tsv')
```

Do we have missing values, and if so, in which columns?
```{r}
sum(is.na(genera_counts))
```
We don't have missing data values. Now we need to know what the different bacterial classes are.

```{r}
column_names <- colnames(genera_counts)

extract_class_from_column_name <- function(column_name) {
  match <- regmatches(column_name, regexec("c__([A-Za-z0-9_-]+)", column_name))
  
  if (length(match[[1]]) > 1) {
    return(match[[1]][2])  
  }
  return(NA)  # No match found
}

classes <- sapply(column_names, extract_class_from_column_name)

unique_classes <- unique(classes)

cat("Unique classes in the dataset:\n")
print(unique_classes)
```
We observe a total of 275 unique classes, excluding the first generated output, which is treated as NA because it originates from the Subject column.


Now we are going to replace all bacterial column names with the bacterial class names:
 
```{r}
column_names <- colnames(genera_counts)

# Function to extract the class from the column name
extract_class_from_column_name <- function(column_name) {
  match <- regmatches(column_name, regexec("c__([A-Za-z0-9_-]+)", column_name))
  
  if (length(match[[1]]) > 1) {
    return(match[[1]][2])  # Return the class part
  }
  return(NA)  # No match found
}

# Obtain a vector of classes for each column
classes <- sapply(column_names, extract_class_from_column_name)

# Remove NAs from unique classes
unique_classes <- unique(classes[!is.na(classes)])

cat("Unique classes in the dataset:\n")
print(unique_classes)

# Function to rename column names without indexes
rename_columns_no_index <- function(column_names, classes) {
  new_column_names <- character(length(column_names))
  
  # Loop over the column names
  for (i in seq_along(column_names)) {
    class_name <- classes[i]
    
    # Only continue if the class_name is valid (not NA)
    if (!is.na(class_name)) {
      new_column_names[i] <- class_name  # No index added, just class name
    } else {
      new_column_names[i] <- "Unknown"  # If no class is found, call the column "Unknown"
    }
  }
  
  return(new_column_names)
}

# Apply function to rename column names without indexes
new_column_names <- rename_columns_no_index(column_names, classes)

# Rename the columns in the dataset
colnames(genera_counts) <- new_column_names

# Check out the updated column names
print(colnames(genera_counts))
head(genera_counts)
```	
Samples column is currently labeled as 'Unknown'. Check if any other column is labeled as 'Unknown' as well.

_hier nog code plaken._

Now, merge the columns with the same name. Additionally, rename the 'Unknown' column back to 'Samples'.
```{r}
# Step 1: Keep ‘Unknown’ untouched and rename to ‘Sample’
genera_counts$Sample <- genera_counts$Unknown
genera_counts <- genera_counts[, names(genera_counts) != "Unknown"]

# Step 2: Check and convert only columns that are numeric
genera_counts_numeric <- genera_counts
genera_counts_numeric[] <- lapply(genera_counts_numeric, function(col) {
  if (is.numeric(col)) {
    col # If the column is already numeric, leave it untouched
  } else if (is.factor(col) || is.character(col)) {
    suppressWarnings(as.numeric(col)) # Try converting text/factor to numeric
  } else {
    NULL # Exclude columns that cannot be converted
  }
})

# Step 3: Combine columns with the same name by merging them together
genera_counts_combined <- as.data.frame(sapply(unique(names(genera_counts_numeric)), function(col) {
  if (col %in% names(genera_counts_numeric)) {
    rowSums(genera_counts_numeric[names(genera_counts_numeric) == col], na.rm = TRUE)
  } else {
    NULL
  }
}))

# Step 4: Add the ‘Sample’ column back to the merged dataset
genera_counts_combined$Sample <- genera_counts$Sample

# Step 5: Move the ‘Sample’ column to the first position
genera_counts_combined <- genera_counts_combined[, c("Sample", setdiff(names(genera_counts_combined), "Sample"))]

# Step 6: Check the result
print(genera_counts_combined)
```
Append the 'Study.Group' column to the 'genera_counts_combined' dataset.

```{r}
genera_counts_combined <- merge(genera_counts_combined, metadata[, c("Sample", "Study.Group")], by = "Sample")
```
Append the 'Subject' column to the 'genera_counts_combined' dataset.

```{r}
genera_counts_combined <- merge(genera_counts_combined, metadata[, c("Sample", "Subject")], by = "Sample")
```


# Machine learning:


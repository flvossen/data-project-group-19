# PG19-group-project-MICROBIOME

# Research question 1 (univariate analysis): 

## Can calprotectin levels be used as a diagnostic biomarker to differentiate between non-IBD, ulcerative colitis (UC), and Crohn's disease (CD) in patients?

**Null hypothesis:** There is no significant difference in fecal calprotectin levels between patients with CD (Crohn's Disease), UC (Ulcerative Colitis), and non-IBD (individuals without inflammatory bowel diseases).

**Alternative hypothesis:** There is a significant difference in fecal calprotectin levels between patients with CD (Crohn's Disease), UC (Ulcerative Colitis), and non-IBD (individuals without inflammatory bowel diseases).

### Data preparation:

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

_hoe hebben we deze NA's gezien en waarom enkel in fecalcal verwijderen?_
Fecal calprotectin contains NA values, we will remove them.
```{r}
clean_data1 <- metadata %>% filter(!is.na(fecalcal)) 
print(clean_data1) 
```
Checking if all the NA values are removed.
```{r}
sum(is.na(clean_data1$fecalcal)) 
```
It is important to examine how the NA values are distributed across the three study groups. This helps to understand whether the missing data is randomly distributed or if there are systematic patterns. For example, if the NA values are concentrated in specific study groups, it may indicate issues in data collection or bias, which could impact the validity of further analyses
```{r} 
na_distribution <- metadata %>% 
group_by(Study.Group) %>%  
summarise( 
  Total_NA = sum(is.na(fecalcal)), 
  Percentage_NA = (Total_NA / n()) * 100
) 
print(na_distribution) 
``` 
We have examined the distribution of missing values, and percentage-wise, there is no notable difference between the three groups.

Visualising the distribution:
```{r} 
ggplot(na_distribution, aes(x = Study.Group, y = Percentage_NA, fill = Study.Group)) + geom_bar(stat = "identity") +
labs( 
  title = "Percentage of Missing Values per Study Group",
  x = "Study Group", 
  y = "Percentage of NA" 
) + 
theme_minimal() 
```
There is no real difference between the groups. Next step is to conduct a statistical test to determine if there is indeed no significant difference in the NA distribution across the study groups. 

Since we are working with categorical variables and need to examine the association between them, we will use a chi-square test, which assumes that the distribution between the groups is random.

The goal is actually to prove the null hypothesis, not to reject it: p > 0.05.
```{r} 
chisq_test_data <- metadata %>% 
  mutate(fecalcal_missing = ifelse(is.na(fecalcal), "Missing", "Not Missing")) %>% 
  count(Study.Group, fecalcal_missing) %>% 
  pivot_wider(names_from = fecalcal_missing, values_from = n, values_fill = 0) 

chisq_result <- chisq.test(chisq_test_data[,-1])  #Disregard the Study.Group column
print(chisq_result) 
```

#### Interpretation: 
Based on the p-value (0.8778), we cannot reject the null hypothesis (H₀). This means that there is no evidence of a significant difference in the number of missing values (NAs) between the different study groups (CD, UC, non-IBD).

The missing values appear to be randomly distributed across the study groups. Therefore, in this case, you can assume that the missing data is likely missing completely at random (MCAR), meaning that the distribution of NAs does not show any systematic pattern and does not introduce bias into your analysis.

Now that the NA distribution has been checked, it is also important to verify whether there are any duplicate subjects, as these could potentially influence the results.
```{r} 
duplicated_subjects <- clean_data[duplicated(clean_data$Subject), ]
```
_waarom staat er in de code hierboven een komma voor ]?_

Retrieving the duplicate subjects:
```{r} 
cat("Aantal dubbele subjects:", nrow(duplicated_subjects), "\n")
print(duplicated_subjects) 
```

Duplicate subjects are removed, and the mean of the numerical data for these subjects is calculated while ignoring the NA values.
```{r} 
metadata_clean <- clean_data %>% 
  group_by(Subject) %>% 
  filter(n_distinct(Study.Group) == 1) %>% 
  summarise( 
    Study.Group = first(Study.Group), 
    fecalcal_mean = mean(fecalcal, na.rm = TRUE), 
    .groups = "drop" 
  )
print(metadata_clean) 
```

Controlling if the duplicate subjects have been removed:
```{r}
duplicated_subjects1 <- metadata_clean[duplicated(metadata_clean$Subject), ] 
cat("dubble subjects:", nrow(duplicated_subjects1), "\n") 
print(duplicated_subjects1) 
```
_again, waarom staat er in de code hierboven een komma voor ]?_

### Descriptive statistics:
_is deze tussentitel een goede vertaling voor 'Beschrijvende statistieken?_
Calculates the descriptive statistics (mean, median, SD, min, max) of fecalcal_mean per Study.Group.
```{r} 
desc_stats <- metadata_clean %>% 
  group_by(Study.Group) %>% 
  summarize( 
    Mean = mean(fecalcal_mean), 
    Median = median(fecalcal_mean), 
    SD = sd(fecalcal_mean), 
    Min = min(fecalcal_mean), 
    Max = max(fecalcal_mean) 
  ) 
print(desc_stats) 
```

#### Interpretation:
We observe that the difference in the average values between CD and UC is relatively small. However, there is a much larger difference in the averages between both CD and non-IBD, as well as UC and non-IBD. Additionally, the minimum values across the study groups are very similar. In contrast, the maximum values are more spread out, except for between UC and CD. Notably, the maximum value for non-IBD is close to the median values of both UC and CD

Visualisation of the constribution:
```{r} 
ggplot(metadata_clean, aes(x = Study.Group, y = fecalcal_mean, fill = Study.Group)) + 
  geom_boxplot() + 
  geom_jitter(width = 0.2, alpha = 0.5) + 
  labs(title = "Calprotectine per Study group", y = "Calprotectine (µg/g)", x =      "Group") + 
  theme_minimal() 
``` 

Histogram of fecal calprotectine per group WITH outliers:
```{r} 
ggplot(metadata_clean, aes(x = fecalcal_mean, fill = Study.Group)) + 
  geom_histogram(binwidth = 50, alpha = 0.6, position = "identity") + 
  facet_wrap(~Study.Group) +  # Create separate plots for each group
  labs(title = "Histogram per Group", x = "Calprotectine (µg/g)", y = "Frequency") + 
  theme_minimal() 
``` 

Calculating the IQR boundaries per group:
```{r} 
quartiles <- metadata_clean %>% 
  group_by(Study.Group) %>% 
  summarize( 
    Q1 = quantile(fecalcal_mean, 0.25), 
    Q3 = quantile(fecalcal_mean, 0.75) 
  ) %>% 
  mutate( 
    IQR = Q3 - Q1, 
    Lower_Bound = Q1 - 1.5 * IQR, 
    Upper_Bound = Q3 + 1.5 * IQR 
  ) 
```

_in de word werd nu nomaals de berekening van de beschrijvende statistieken gedaan (mean,median,SD,min,max) wat dan exact hetzelfde resultaat geeft als hierboven, dus dit heb ik weggelaten_

Add bounderaries to the data:
```{r} 
clean_data_unique <- metadata_clean %>% 
  left_join(quartiles, by = "Study.Group") %>% 
  filter(fecalcal_mean>= Lower_Bound & fecalcal_mean <= Upper_Bound) 
```

Boxplot of fecal calprotectine per group WITHOUT outliers:
```{r} 
ggplot(clean_data_unique, aes(x = Study.Group, y = fecalcal_mean, fill = Study.Group)) + 
  geom_boxplot() + 
  geom_jitter(width = 0.2, alpha = 0.5) + 
  labs(title = "Fecalcal per Group (no outliers)", y = "Fecalcal (µg/g)", x = "Group") + 
  theme_minimal()
``` 

Histogram of fecal calprotectine per group WITHOUT outliers:
```{r} 
ggplot(clean_data_unique, aes(x = fecalcal_mean, fill = Study.Group)) + 
  geom_histogram(binwidth = 50, alpha = 0.6, position = "identity") + 
  facet_wrap(~Study.Group) +  
  labs(title = "Histogram per Group (no outliers)", x = "Calprotectine (µg/g)", y = "Frequency") + 
  theme_minimal()
```

#### Interpretation:
We have added 5 new variables to the clean_data_unique dataset:
- Q1: The first quartile (25th percentile).
- Q3: The third quartile (75th percentile).
- IQR: The difference between Q3 and Q1.
- Lower Bound = Q1 − 1.5 ⋅ IQR
- Upper Bound = Q3 + 1.5 ⋅ IQR

After removing outliers, we are left with 78 subjects. In the histogram per group (without outliers) of clean_data_unique, we notice that the data do not follow a normal distribution, as no bell-shaped curve is observed per study group. Therefore, we will proceed with a Kruskal-Wallis test to analyze the differences between the groups.

Kruskal-Wallis test on the clean_data_unique (without outliers and duplicate subjects).
```{r} 
install.packages("coin") 
library(coin) 
kruskal.test(fecalcal_mean ~ Study.Group, data = clean_data_unique) 
``` 

#### Interpretation:
There is a significant difference in fecal calprotectine levels across the three different study groups, with a standard asymptotic p-value of 0.0001005. We will now first calculate the exact p-value and then examine which specific groups differ using the Wilcoxon test as a post hoc analysis.

While the standard asymptotic p-value may not be optimal with a small number of observations, we have 77 observations in this case, making the p-value reliable. Therefore, we will proceed with calculating the exact p-value.

#### Conclusion:
We can conclude that there is an extremely significant difference (p = 0.0001005) in the distribution of calprotectin concentration due to the three study groups.

Post-hoc analysis will be conducted using the pairwise Wilcoxon test to examine the differences between the groups.
```{r} 
pairwise_wilcox <- pairwise.wilcox.test( 
  x = clean_data_unique$fecalcal_mean, 
  g = clean_data_unique$Study.Group, 
  p.adjust.method = "bonferroni"  # Correction for multiple testing 
) 
print(pairwise_wilcox) 
``` 
#### Interpretation:
The output displays the p-values from pairwise comparisons in a matrix format, with a Bonferroni correction applied for multiple testing. The results indicate that both CD and UC show significant differences in fecal calprotectin levels compared to non-IBD (p = 0.00014 and p = 0.00027, respectively). However, there is no significant difference in fecal calprotectin levels between CD and UC (p = 1.0000), meaning that fecal calprotectin cannot be used to distinguish between these two groups.

#### Conclusion:
Fecal calprotectin can effectively differentiate non-IBD from both UC and CD, but it is not a reliable marker to distinguish between CD and UC.

Creating a boxplot using ggboxplot, with Wilcoxon p-values added to show statistical significance.
```{r} 
ggboxplot( 
  clean_data_unique,  
  x = "Study.Group",  
  y = "fecalcal_mean",  
  fill = "Study.Group",  
  add = "jitter",  # Add points to show data distribution 
  palette = "jco"  # Choose a color scheme 
) + 
  stat_compare_means( 
    method = "kruskal.test",  # Kruskal-Wallis for global comparison 
    label.y = max(clean_data_unique$fecalcal_mean) * 1.1  # Position global p-value 
  ) + 
  stat_compare_means( 
    comparisons = list(c("CD", "UC"), c("CD", "non-IBD"), c("UC", "non-IBD")),  # Comparisons 
    method = "wilcox.test",  
    p.adjust.method = "bonferroni",  # Correction for multiple testing 
    label = "p.signif"  # Use *, **, *** for p-values 
  ) + 
  labs( 
    title = "Calprotectin Levels per Study Group", 
    x = "Group", 
    y = "Calprotectin (µg/g)" 
  )
``` 

#### Interpretation: 
Since UC and CD do not show a significant difference in fecal calprotectine levels, they can be grouped together as IBD (Inflammatory Bowel Disease) and compared to non-IBD. This allows for a broader comparison between IBD and non-IBD based on fecalcal levels."

```{r} 
install.packages("dplyr") 
library(dplyr) 
```

Creating a new column where UC and CD are combined into IBD:
```{r} 
clean_data_unique <- clean_data_unique %>% 
  mutate(Group_Combined = ifelse(Study.Group %in% c("UC", "CD"), "IBD", Study.Group)) 
``` 

Kruskal-Wallis Test for IBD vs Non-IBD:
```{r} 
install.packages("ggpubr") 
library(ggpubr) 
```
```{r} 
kruskal_combined <- kruskal.test(fecalcal_mean ~ Group_Combined, data = clean_data_unique) 
print(kruskal_combined) 
```

#### Interpretation:
The comparison between the non-IBD and IBD groups is statistically significant (p = 1.79e-5 < 0.05), suggesting that fecal calprotectin levels can effectively distinguish between these groups.

For the confidence intervals, it is preferable to use the median since our data is not normally distributed. The median is much more robust to outliers than the mean.

```{r} 
install.packages("Hmisc") 
library(Hmisc)
set.seed(123)  # For reproducibility
library(dplyr)
```

Perform bootstrapping to calculate the confidence intervals for the median of each group:
```{r} 
bootstrap_ci <- function(data, group, value, n_boot = 1000, conf_level = 0.95) { 
  groups <- unique(data[[group]]) 
  results <- data.frame(Group = character(0), CI_Lower = numeric(0), CI_Upper = numeric(0)) 
  
  for (g in groups) { 
    group_data <- data %>% filter(!!sym(group) == g) 
    medians <- replicate(n_boot, median(sample(group_data[[value]], replace = TRUE))) 
    lower <- quantile(medians, (1 - conf_level) / 2) 
    upper <- quantile(medians, 1 - (1 - conf_level) / 2) 
    results <- rbind(results, data.frame(Group = g, CI_Lower = lower, CI_Upper = upper)) 
  } 
  
  return(results) 
}
```

Calculate the 95% confidence intervals for the median of the fecalcal_mean variable for each group:
```{r} 
ci_results <- bootstrap_ci( 
  data = clean_data_unique, 
  group = "Group_Combined", 
  value = "fecalcal_mean", 
  n_boot = 1000, 
  conf_level = 0.95 
)

print(ci_results)
```

#### Interpretation:
The 95% confidence intervals (CIs) for the median fecal calprotectin levels in the IBD and non-IBD groups show distinct patterns. For the IBD group, the CI ranges from 41.48 µg/g to 147.69 µg/g, indicating that the median calprotectin level is significantly elevated, with some variability in the data. In contrast, the non-IBD group has a narrower CI, from 14.87 µg/g to 22.72 µg/g, suggesting a more consistent distribution and much lower median levels compared to the IBD group.

Crucially, the non-overlapping CIs between these groups indicate a statistically significant difference in fecal calprotectin levels. This finding strongly supports the use of fecal calprotectin as a reliable biomarker to distinguish between individuals with IBD and those without. The elevated levels in the IBD group reflect underlying inflammation, further underscoring its diagnostic relevance.

The Kruskal-Wallis test is used to determine if there is a significant difference in the median fecal calprotectin levels between the "IBD" and "non-IBD" groups:
```{r} 
kruskal_combined <- kruskal.test(fecalcal_mean ~ Group_Combined, data = clean_data_unique) 
```

Calculating the effect size (eta-squared, η²):
```{r}
H <- kruskal_combined$statistic
k <- length(unique(clean_data_unique$Group_Combined))
n <- nrow(clean_data_unique)

# Calculate eta-squared
eta_squared <- (H - k + 1) / (n - k)
print(eta_squared)
```

#### Interpretation:
The output provides the effect size (η²) of the Kruskal-Wallis test. A value of 0.229 (rounded) indicates that approximately 22.9% of the variance in calprotectin levels can be explained by the differences between the groups (IBD vs non-IBD). This suggests that the group classification has a notable impact on the variation observed in calprotectin levels.

#### Conclusion:
The effect size indicates that the group classification ("IBD" vs "non-IBD") significantly contributes to the variation in fecal calprotectin levels, further supporting the utility of fecal calprotectin as a biomarker for distinguishing between these two groups.

---------------------------------------------------------------------------------------

# Research question 2 (multivariate analysis): 

## Are bacterial classes associated with Non-IBD, Ulcerative Colitis (UC), and Crohn's Disease (CD) in terms of microbiome composition?

**Null hypothesis:** There is no significant difference in bacterial composition between the three study groups (class CD, UC and non-IBD).

**Alternative hypothesis:** There is significant difference in bacterial composition between the three study groups (class CD, UC and non-IBD).

### Data preparation:

Read data
```{r}
genera_counts <- read_tsv('genera.counts.tsv')
```

Check for missing values.
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

# Obtain a vector of classes for each column
classes <- sapply(column_names, extract_class_from_column_name)

unique_classes <- unique(classes)

cat("Unique classes in the dataset:\n")
print(unique_classes)
```
We observe a total of 275 unique classes, excluding the first generated output, which is treated as NA because it originates from the Sample column.


Now we are going to replace all bacterial column names with the bacterial class names:
 
```{r}
# Function to rename column names
rename_columns <- function(column_names, classes) {
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

# Apply function to rename column names
new_column_names <- rename_columns(column_names, classes)

# Rename the columns in the dataset
colnames(genera_counts) <- new_column_names

# Check out the updated column names
print(colnames(genera_counts))
head(genera_counts)
```	


Samples column is currently labeled as 'Unknown'. Check if any other column is labeled as 'Unknown' as well.


```{r}
# Count how many times ‘Unknown’ appears in the column names
unknown_count <- sum(colnames(genera_counts) == "Unknown")

# Print the result
cat("Aantal 'Unknown' kolommen:", unknown_count, "\n")
```

Now, merge the columns with the same name. Only the column 'Sample' was named 'Unknown'. So, rename the 'Unknown' column back to 'Sample'.
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
Append the 'Study.Group' column from the dataset 'metadata' to the 'genera_counts_combined' dataset.

```{r}
genera_counts_combined <- merge(genera_counts_combined, metadata[, c("Sample", "Study.Group")], by = "Sample")
```
Append the 'Subject' column from the dataset 'metadata' to the 'genera_counts_combined' dataset.

```{r}
genera_counts_combined <- merge(genera_counts_combined, metadata[, c("Sample", "Subject")], by = "Sample")
```

Check the number of duplicate subjects.
```{r}
# Check for duplicate values in the ‘Subject’ column
duplicated_subjects <- genera_counts_combined[duplicated(genera_counts_combined$Subject), ]

# Print the number of duplicate values
cat("Amount of duplicate subjects:", nrow(duplicated_subjects), "\n")

# View duplicate values
print(duplicated_subjects)
```
The dataset contains 277 duplicate subjects.

Now take the average of the double subjects.
```{r}
# Calculate the average for Subjects and retain Study.Group
genera_counts_combined_clean <- genera_counts_combined %>%
  group_by(Subject) %>%
  summarise(
    Study.Group = first(Study.Group), 
    across(where(is.numeric), mean, na.rm = TRUE),
    .groups = "drop"
  )

# View the resulting dataset
print(genera_counts_combined_clean)

# Check the number of unique subjects
cat("Amount of unique subjects:", n_distinct(genera_counts_combined_clean$Subject), "\n")
```
We now have 105 unique subjects.


### visualisation

_nog alles komen van correlatiematrix_

### Dimensionality Reduction::

```{r}
# Step 1: Perform the PCA on the numeric columns
bacteria_data <- genera_counts_combined_clean %>%
  select(-c(Subject, Study.Group))  # Remove the non-numeric columns

# Run PCA
pca_result <- prcomp(bacteria_data)

# Step 2: Extract the first two components for visualisation
pca_data <- as.data.frame(pca_result$x)
pca_data$Study.Group <- genera_counts_combined_clean$Study.Group

# Step 3: Visualise PCA with ggplot2
library(ggplot2)
ggplot(pca_data, aes(x = PC1, y = PC2, color = Study.Group)) +
  geom_point(size = 3) +
  labs(title = "PCA of bacterial distribution",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal()

```
The code aims to reduce the dimensionality of the bacterial abundance data using PCA, and then visualize the data in two dimensions (PC1 and PC2) to see how different study groups (UC, CD, non-IBD) cluster or separate from each other based on their bacterial profiles.

Check for possible outliers:
```{r}
# Calculate Mahalanobis distance in head space
pca_scores <- pca_result$x[, 1:2]  # The first two main components
mahalanobis_pca <- mahalanobis(pca_scores, colMeans(pca_scores), cov(pca_scores))

# Add the distance to your data frame
genera_counts_combined_clean$mahalanobis_pca <- mahalanobis_pca

# Threshold for identifying outliers
threshold_pca <- qchisq(0.95, df = 2)  # 95% threshold, 2 dimensions
outliers_pca <- which(mahalanobis_pca > threshold_pca)

# Visualise the outliers
plot(pca_scores, col = ifelse(mahalanobis_pca > threshold_pca, "red", "black"))
```
We have decided to retain the outliers. Retaining outliers in PCA allows for capturing important, rare variations and reflecting the true complexity and diversity of real-world data, ensuring that meaningful patterns and extreme but legitimate variations are not lost.

We used a screeplot to identify the optimal number of principal components to retain by showing the variance explained by each component and highlighting the point where additional components contribute less.
```{r}
screeplot (pca_result, type='lines',main="PC Variance by PC # (Screeplot) ")
abline (h=mean ( (pca_result$sdev)^2), col= 'gray' , lty=2)
legend ("right", "Mean Variance" ,lty=2, col='gray',bty='n')
cumulative_variance <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
num_pcs <- which(cumulative_variance >= 0.9)[1]  # First PC to explain ≥90%
print(num_pcs)
```
Look for the "elbow point," where the additional variance explained by subsequent principal components becomes minimal, and select the components that together explain more than 90% of the total variance, here PC2.

```{r}
# Example: Extract loadings for the principal components
pca_loadings <- as.data.frame(pca_result$rotation)
most_influential_classes <- rownames(pca_loadings[order(abs(pca_loadings$PC2), decreasing = TRUE)[1:5], ])
```
From this, we can identify which classes play the most important role and use them to examine whether these differ across study groups. Taking the highest variables (or loadings) of a principal component, means identifying the original variables that contribute most significantly to that component. These variables have the strongest influence on the direction and variance explained by the component, highlighting the key factors or features driving the patterns captured by that principal component. Here we used principal component 2. 

```{r}
# PCA loadings as a data frame
pca_loadings <- as.data.frame(pca_result$rotation)

# Find the top 5 most influential features for each PC
top_influential_classes <- lapply(names(pca_loadings), function(pc) {
  # Extract the current PC's loadings
  current_pc <- pca_loadings[[pc]]
  
  # Find the indices of the top 5 absolute loadings
  top_indices <- order(abs(current_pc), decreasing = TRUE)[1:5]
  
  # Return the feature names and their loadings
  data.frame(Feature = rownames(pca_loadings)[top_indices],
             Loading = current_pc[top_indices],
             PC = pc)
})

# Combine results into a single data frame
top_influential_classes <- do.call(rbind, top_influential_classes)

# View the result
print(top_influential_classes)
```
The top 5 bacterial classes are; Clostridia, Bacteroidia, Gammaproteobacteria, Negativicutes, and Bacilli.

### Descriptive Statistics:

Calculating the mean, median, and standard deviation:
```{r}
# Include Study.Group in the bacteria_data subset
bacteria_data <- genera_counts_combined_clean[, c("Clostridia", "Bacteroidia", "Gammaproteobacteria", "Negativicutes", "Bacilli", "Study.Group")]

# Calculate mean, median, and standard deviation per bacterial class grouped by Study.Group
mean_median_sd <- bacteria_data %>%
  group_by(Study.Group) %>%
  summarise(across(
    where(is.numeric),
    list(mean = mean, median = median, sd = sd),
    na.rm = TRUE
  ))

# Print the results
print(mean_median_sd)
```
_nog iets typen over de median, mean and sd._

### Normality Testing:
To determine if the five bacterial classes are normally distributed, we created histograms for each bacterial class by study group and conducted a Shapiro-Wilk test.
```{r}
# List of bacteria classes
bacterial_classes <- c("Clostridia", "Bacteroidia", "Gammaproteobacteria", "Negativicutes", "Bacilli")

# For each bacterial class, create histograms by Study.Group
for (class in bacterial_classes) {
  plot <- ggplot(genera_counts_combined_clean, aes_string(x = class, fill = "Study.Group")) +
    geom_histogram(bins = 30, alpha = 0.7, position = "dodge", color = "black") +
    facet_wrap(~Study.Group, scales = "free") +  # Put facets per Study.Group
    labs(
      title = paste("Histogram of", class, "per Study Group"),
      x = class,
      y = "Frequency"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))  # Title centred
  print(plot)
}

# Shapiro-Wilk test for each bacterial class
shapiro_results <- sapply(genera_counts_combined_clean[bacterial_classes], function(x) shapiro.test(x)$p.value)

# Display results
shapiro_results
```
If the p-value of the Shapiro-Wilk test is greater than 0.05, the data can be considered normally distributed; otherwise, a p-value below 0.05 indicates significant deviation from normality. In this case, only one value exceeds 0.05, suggesting that most of the data significantly deviate from a normal distribution.

### PERMANOVA:
```{r}
library(vegan)  # For adonis function

bacteria_data <- genera_counts_combined_clean[, c("Clostridia", "Bacteroidia", "Gammaproteobacteria", "Negativicutes", "Bacilli")]
study_group <- genera_counts_combined_clean$Study.Group

# Perform PERMANOVA
perm_result <- adonis2(bacteria_data ~ study_group , method = "euclidean", permutations = 999)
print(perm_result)
```
#### Interpretation PERMANOVA:
R2 is the proportion of total variation explained by the study group factor is 0.01432. This means that only about 1.43% of the total variation in bacterial composition can be explained by the study group classification.

The F-statistic is 0.7408, , which measures the ratio of between-group variation to within-group variation. A higher F-value suggests more separation between groups, but this value is relatively low here.

The p-value is 0.52, which is greater than 0.05, indicating that there is no significant difference in the bacterial compositions between the study groups based on the Euclidean distance. 

### Conclusion:
The PERMANOVA results indicate that the null hypothesis cannot be rejected. This suggests:
- No significant association exists between the bacterial composition and study groups (Non-IBD, UC, CD).
- The small R² value highlights that only a tiny fraction of variance in bacterial composition is attributable to the study group factor.



# Machine learning:


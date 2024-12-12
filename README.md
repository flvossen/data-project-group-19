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
  return(NA)  # Geen match gevonden
}

classes <- sapply(column_names, extract_class_from_column_name)

unique_classes <- unique(classes)

cat("Unieke classes in de dataset:\n")
print(unique_classes)
```
In this, we see that we have 275 unique classes (the first class of output is counted as an NA). 


Now we are going to replace all bacterial column names with the bacterial class name:
 
```
column_names <- colnames(genera_counts)

# Functie om de klasse te extraheren uit de kolomnaam
extract_class_from_column_name <- function(column_name) {
  match <- regmatches(column_name, regexec("c__([A-Za-z0-9_-]+)", column_name))
  
  if (length(match[[1]]) > 1) {
    return(match[[1]][2])  # Return the class part
  }
  return(NA)  # No match found
}

# Verkrijg een vector van klassen voor elke kolom
classes <- sapply(column_names, extract_class_from_column_name)

# Verwijder NA's uit de unieke klassen
unique_classes <- unique(classes[!is.na(classes)])

cat("Unieke classes in de dataset:\n")
print(unique_classes)

# Functie om de kolomnamen te hernoemen zonder indexen
rename_columns_no_index <- function(column_names, classes) {
  new_column_names <- character(length(column_names))
  
  # Loop over de kolomnamen
  for (i in seq_along(column_names)) {
    class_name <- classes[i]
    
    # Alleen doorgaan als de class_name geldig is (niet NA)
    if (!is.na(class_name)) {
      new_column_names[i] <- class_name  # No index added, just class name
    } else {
      new_column_names[i] <- "Unknown"  # Als geen klasse is gevonden, noem de kolom "Unknown"
    }
  }
  
  return(new_column_names)
}

# Pas de functie toe om kolomnamen zonder indexen te hernoemen
new_column_names <- rename_columns_no_index(column_names, classes)

# Hernoem de kolommen in de dataset
colnames(genera_counts) <- new_column_names

# Bekijk de vernieuwde kolomnamen
print(colnames(genera_counts))
head(genera_counts)
```	
Sample is currently labeled as 'unknown'. Check if any other column is labeled as 'unknown' as well.

_hier nog code plaken._

Now, merge the columns with the same name. Additionally, rename the 'unknown' column back to 'samples'.
```
# Stap 1: Houd "Unknown" ongemoeid en hernoem naar "Sample"
genera_counts$Sample <- genera_counts$Unknown
genera_counts <- genera_counts[, names(genera_counts) != "Unknown"]

# Stap 2: Controleer en converteer alleen kolommen die numeriek zijn
genera_counts_numeric <- genera_counts
genera_counts_numeric[] <- lapply(genera_counts_numeric, function(col) {
  if (is.numeric(col)) {
    col # Als de kolom al numeriek is, laat deze ongemoeid
  } else if (is.factor(col) || is.character(col)) {
    suppressWarnings(as.numeric(col)) # Probeer tekst/factor om te zetten naar numeriek
  } else {
    NULL # Kolommen die niet kunnen worden geconverteerd, uitsluiten
  }
})

# Stap 3: Combineer kolommen met dezelfde naam door ze samen te voegen
genera_counts_combined <- as.data.frame(sapply(unique(names(genera_counts_numeric)), function(col) {
  if (col %in% names(genera_counts_numeric)) {
    rowSums(genera_counts_numeric[names(genera_counts_numeric) == col], na.rm = TRUE)
  } else {
    NULL
  }
}))

# Stap 4: Voeg de "Sample"-kolom terug toe aan de samengevoegde dataset
genera_counts_combined$Sample <- genera_counts$Sample

# Stap 5: Verplaats de "Sample"-kolom naar de eerste positie (optioneel)
genera_counts_combined <- genera_counts_combined[, c("Sample", setdiff(names(genera_counts_combined), "Sample"))]

# Stap 6: Controleer het resultaat
print(genera_counts_combined)
```
Append the 'Study.Group' column to the 'genera_counts_combined' dataset.

```
genera_counts_combined <- merge(genera_counts_combined, metadata[, c("Sample", "Study.Group")], by = "Sample")
```
Append the 'Subject' column to the 'genera_counts_combined' dataset.

```
genera_counts_combined <- merge(genera_counts_combined, metadata[, c("Sample", "Subject")], by = "Sample")
```


# Machine learning:


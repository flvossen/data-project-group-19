# data-project-group-19

## Research question 1:





## Research question 2:


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

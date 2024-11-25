# Questionnaire Statistics 
# 20 participants (F=10, M=10) 

library(readr)

data <- read.csv("~/Desktop/Data/STATS/ISVIQuest.csv")
View(data)

# Load relevant packages 

library(tidyverse)
library(ggplot2)
library(lsr)
library(car)
library(dplyr)

  
# Mean Age
  
mean(data$What.is.your.age...Please.write.your.answer.in.numerical.values.)
# 27.45

sd(data$What.is.your.age...Please.write.your.answer.in.numerical.values.)
# 2.96 

# Left handed 
# 5 participants

## VVIQ QUESTIONNAIRE
# Quantifies the intensity to which people can visualise settings, persons and objects in mind. 
# 16 items clustered in four groups. 
# Mental image rated along a 5-point libert scale where 5 indicated "perfectly clear and as vivid as normal vision"
# No image at all - rated 1 
# The most score is 80 for 16 questions, rated at most 1-5. 


# Separate out data into VVIQ  by participant number

VVIQ <- data[,2:23]
View(VVIQ)

# Define a character vector of participant IDs
participant_ids <- c("01", "02", "04", "03", "05", "07", "06", "08", "09", "11",
                     "10", "12", "14", "13", "15", "18", "17", "16", "19", "20")

# Assign the participant IDs to the dataframe
VVIQ$Participant_ID <- participant_ids

# View the updated dataframe
View(VVIQ)

# Reorder the columns of the dataframe
VVIQ <- VVIQ[, c("Participant_ID", setdiff(names(VVIQ), "Participant_ID"))]

View(VVIQ)


## Rename column names of VVIQ data set 
colnames(VVIQ) <- c('ID', 'Unique_code', 'Gender','Age', 'Handedness', 'English', 'Disorders','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16')
View(VVIQ)

class(VVIQ$Participant_ID)  


# Define the mapping of values to replacements
value_mapping <- c("No image at all (only \"knowing\" that you are thinking of the object)" = 1,
                   "Vague, and dim" = 2,
                   "Moderately clear and vivid" = 3,
                   "Clear and reasonably vivid" = 4,
                   "Perfectly clear and as vivid as normal vision" = 5)


# Apply the mapping to columns 8 to 23 in the VVIQ dataframe

VVIQ_1 <- VVIQ %>%
  mutate_at(vars(Q1:Q16), ~ ifelse(. %in% names(value_mapping), value_mapping[.], .))

View(VVIQ_1)

# Calculate the total sum of VVIQ scores
sum_totals <- rowSums(VVIQ_1[, 8:23], na.rm = TRUE)

# Add the sum totals as a new column to the dataframe
VVIQ_1$Sum_Total <- sum_totals


# Calculate the mean scores for each participant (across rows)
mean_scores <- rowMeans(VVIQ_1[, 8:23], na.rm = TRUE)

# Add the mean scores as a new column to the dataframe
VVIQ_1$Mean_Score <- mean_scores
View(VVIQ_1)


# Reorder the VVIQ_1 dataframe in descending order based on Sum_Total scores
ordered_VVIQ <- VVIQ_1 %>%
  arrange(desc(Sum_Total))

# Print the ordered dataframe
View(ordered_VVIQ)

# Statistics of overall scores

mean(ordered_VVIQ$Sum_Total) 
# 52.6

median(ordered_VVIQ$Sum_Total)
# 53.5

sd(ordered_VVIQ$Sum_Total)
# 15.31219

# Save the ordered dataframe as a CSV file 

#write.csv(ordered_VVIQ, "Ordered_VVIQ.csv", row.names = FALSE)

# Plot the distribution

#hist(ordered_VVIQ$Sum_Total, main = "Distribution of VVIQ scores", xlab = "VVIQ scores", ylab= 'Count')

hist_plot <- ggplot(ordered_VVIQ, aes(x = Sum_Total)) +
  geom_histogram(binwidth = 5, color = "#b8acd1", alpha=0.6) +
  labs(title = "Distribution of VVIQ scores", x = "VVIQ scores", y = "Count")



# GENERATE LINEAR REGRESSION FOR VVIQ and CNN Classification Accuracy scores 

# Input data column for CNN Accuracy scores
 

ordered_VVIQ$CNNAcc <- c(97,98,93,71,97,78,98,93,94,66,98,86,87,97,69,95,96,94,77,99)
  


#fit linear regression model to dataset and view model summary
VVIQ_CNN <- lm(CNNAcc~Sum_Total, data=ordered_VVIQ)
summary(VVIQ_CNN)


# PLOT Linear Regression with ggplot2

#create regression plot with customized style

VVIQ_CNN <-ggplot(ordered_VVIQ,aes(Sum_Total, CNNAcc)) +
  geom_point() +
  geom_smooth(method='lm', se=TRUE, color='turquoise4') +
  theme_minimal() +
  labs(x='VVIQ Scores', y='Accuracy Scores', title='VVIQ vs. CNN Accuracy') +
  theme(plot.title = element_text(hjust=0.5, size=20, face='bold')) 

ggsave("VVIQ_CNN.png", plot =VVIQ_CNN , width = 6, height = 4)

# Add in accuracies IS 

ordered_VVIQ$ISAcc <-c(96,95,93,65,91,99,97,69,77,97,62,68,61,53,54,88,74,68,65,85)


#Fit linear regression model to VI and IS CNN Accuracy score

VI_IS <- lm(ISAcc~CNNAcc, data=ordered_VVIQ)
summary(VI_IS)

VI_IS <-ggplot(ordered_VVIQ,aes(CNNAcc, ISAcc)) +
  geom_point() +
  geom_smooth(method='lm', se=TRUE, color='turquoise4') +
  theme_minimal() +
  labs(x='VI Accuracy', y='IS Accuracy', title='VI vs IS CNN Accuracy') +
  theme(plot.title = element_text(hjust=0.5, size=20, face='bold')) 



# Perform paired t-test between grouped scores VI and IS
t_result <- t.test(ordered_VVIQ$CNNAcc, ordered_VVIQ$ISAcc, paired = TRUE)

# Print t-test result
print(t_result)

#Result data:  ordered_VVIQ$CNNAcc and ordered_VVIQ$ISAcc
#t = 2.8918, df = 19, p-value = 0.009345
#alternative hypothesis: true mean difference is not equal to 0
#95 percent confidence interval:
#  3.121404 19.478596
#sample estimates:
#  mean difference 
#11.3 


class(ordered_VVIQ$CNNAcc) 
class(ordered_VVIQ$ISAcc) 


mean(ordered_VVIQ$CNNAcc)
#89.15

mean(ordered_VVIQ$ISAcc)
#77.85

# Extract the columns of interest
cnna_cc <- ordered_VVIQ$CNNAcc
is_acc <- ordered_VVIQ$ISAcc

# Perform paired t-test
t_result <- t.test(cnna_cc, is_acc, paired = TRUE)

# Extract t-test statistics and p-values
t_statistic <- t_result$statistic
p_value <- t_result$p.value

print(t_result)

# Create a data frame to store the results
t_test_results <- data.frame(Participant_ID = ordered_VVIQ$ID,
                             t_statistic = t_statistic,
                             p_value = p_value)



## DYSLEXIA QUESTIONNAIRE

## Score less than 45 - probably non-dyslexic: Research results: no individual who was diagnosed as dyslexic through 
# a full assessment was found to have scored less than 45 and therefore it is unlikely that if you score under 45 you will
# be dyslexic.

## Score 45 to 60 - showing signs consistent with mild dyslexia. Research results: most of those who were in this category showed 
# signs of being at least moderately dyslexic. However, a number of persons not previously diagnosed as dyslexic
# (though they could just be unrecognised and undiagnosed) fell into this category.

## Score Greater than 60 - signs consistent with moderate or severe dyslexia.
# Research results: all those who recorded scores of more than 60 were diagnosed as
# moderately or severely dyslexic. Therefore we would suggest that a score greater than 60
# suggests moderate or severe dyslexia. Please note that this should not be regarded as an
# assessment of one’s difficulties. But if you feel that a dyslexia-type problem may exist, further
# advice should be sought.

# Separate out data into Dyslexia questionnaire by particpant number
Dys <- data[,24:38]
View(Dys)

# Create a column for the Participant number for Dyslexia questionnaire

# Define a character vector of participant IDs
participant_ids <- c("01", "02", "04", "03", "05", "07", "06", "08", "09", "11",
                     "10", "12", "14", "13", "15", "18", "17", "16", "19", "20")

# Assign the participant IDs to the dataframe
Dys$Participant_ID <- participant_ids

# View the updated dataframe
View(Dys)

# Reorder the columns of the dataframe
Dys <- Dys[, c("Participant_ID", setdiff(names(Dys), "Participant_ID"))]

View(Dys)

## Rename column names of "Dys" data set 
colnames(Dys) <- c('ID','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15')
View(Dys)


# Q1 Define the mapping of values to replacements
Q1_value_mapping <- c("Rarely - 3" = 3,
                   "Occasionally - 6" = 6,
                   "Often - 9" = 9,
                   "Most of the time - 12" = 12)

# Apply the mapping to Q1 in Dys data frame

Dys_1 <- Dys %>%
  mutate_at(vars(Q1), ~ ifelse(. %in% names(Q1_value_mapping), Q1_value_mapping[.], .))


# Q2 Define the mapping of values to replacements
Q2_value_mapping <- c("Rarely - 2" = 2,
                   "Occasionally - 4" = 4,
                   "Often - 6" = 6,
                   "Most of the time - 8" = 8)

Dys_2 <-Dys_1 %>%
  mutate_at(vars(Q2), ~ ifelse(. %in% names(Q2_value_mapping), Q2_value_mapping[.], .))


# Q3-10 Define the mapping of values to replacements
Q10_value_mapping <- c("Rarely - 1" = 1,
                      "Occasionally - 2" = 2,
                      "Often - 3" = 3,
                      "Most of the time - 4" = 4)

Dys_3 <- Dys_2 %>%
  mutate_at(vars(Q3:Q10), ~ ifelse(. %in% names(Q10_value_mapping), Q10_value_mapping[.], .))


# Q11 Define the mapping of values to replacements
Q11_value_mapping <- c("Easy - 3" = 3,
                      "Challenging - 6" = 6,
                      "Difficult - 9" = 9,
                      "Very Difficult - 12" = 12)

Dys_4 <-Dys_3 %>%
  mutate_at(vars(Q11), ~ ifelse(. %in% names(Q11_value_mapping), Q11_value_mapping[.], .))


# Q12-13 Define the mapping of values to replacements
Q13_value_mapping <- c("Easy - 2" = 2,
                       "Challenging - 4" = 4,
                       "Difficult - 6" = 6,
                       "Very Difficult - 8" = 8)

Dys_5 <- Dys_4 %>%
  mutate_at(vars(Q12:Q13), ~ ifelse(. %in% names(Q13_value_mapping), Q13_value_mapping[.], .))

# Q14-15 Define the mapping of values to replacements
Q15_value_mapping <- c("Easy - 1" = 1,
                       "Challenging - 2" = 2,
                       "Difficult - 3" = 3,
                       "Very Difficult - 4" = 4)

Dys_6 <- Dys_5 %>%
  mutate_at(vars(Q14:Q15), ~ ifelse(. %in% names(Q15_value_mapping), Q15_value_mapping[.], .))

View(Dys_6)



# Calculate the total sum of Dys scores


# Assuming your dataframe is named Dys_6
columns_to_sum <- Dys_6[, 2:16] # Replace with the appropriate column indices

# Convert the columns to numeric
numeric_columns <- as.data.frame(lapply(columns_to_sum, as.numeric))

# Calculate the row sums
sum_totals <- rowSums(numeric_columns, na.rm = TRUE)


# Add the sum totals as a new column to the dataframe
Dys_6$Sum_Total <- sum_totals

View(Dys_6)

# Calculate the mean scores for each participant (across rows)
#mean_scores <- rowMeans(Dys_6[, 2:16], na.rm = TRUE)

# Add the mean scores as a new column to the dataframe
#Dys_6$Mean_Score <- mean_scores


# Reorder the Dys_6 dataframe in descending order based on Sum_Total scores
ordered_Dys <- Dys_6 %>%
  arrange(desc(Sum_Total))

# Print the ordered dataframe
View(ordered_Dys)

# Statistics of overall scores

mean(ordered_Dys$Sum_Total) 
# 30.8

median(ordered_Dys$Sum_Total) 
# 30

sd(ordered_Dys$Sum_Total) 
# 5.66

# Save the ordered dataframe as a CSV file 

write.csv(ordered_Dys, "Ordered_Dys.csv", row.names = FALSE)




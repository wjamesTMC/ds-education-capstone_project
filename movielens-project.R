##############################################################################
#
# Capstone - Movielens Project
# B. James (BCDL)
#
# Description: Create a movie recommendation system using the MovieLens dataset
# and the tools shown throughout the courses. Use the 10M version of the
# MovieLens dataset.
#
# Train a machine learning algorithm using the inputs in one subset to predict
# movie ratings in the validation set. The project will be assessed by peer
# grading.
#
# The submission for the MovieLens project will be three files: a report in the
# form of an Rmd file, a report in the form of a PDF document knit from the Rmd
# file, and an R script or Rmd file that generates the predicted movie ratings
# and calculates RMSE. The grade for the project will be based on two factors:
#
# The report and script (75%) The RMSE returned by testing your algorithm on
# the validation set (25%).
#
##############################################################################

# introduction and Overview
#    The dataset
#    Project goalsand summarizes
#    Overall approach / steps Performed

# Methods and analysis
#    Data clearning
#    data exploration and visualization
#    Insights and modeling approach

# Results

# Conclusions

#----------------------------------------------------------------------------- 
# Set ups, downloads, and establish data sets
#-----------------------------------------------------------------------------

# Necessary libraries

library(tidyverse)
library(tidyr)
library(caret)
library(dplyr)
library(data.table)
library(splitstackshape)
library(reshape2)

# Download the MovieLens data 

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Create the datasets

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% 
         mutate(movieId = as.numeric(levels(movieId))[movieId],
         title = as.character(title),
         genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Create the validation set (10% of MovieLens data)

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
str(edx)

#----------------------------------------------------------------------------- 
# Develop a loss function for model comparisons
#-----------------------------------------------------------------------------

# We define yui as the rating for movie i by user u and y hat ui as our
# prediction. The residual mean squared error is the  error we make when
# predicting a movie rating. This function computes the residual means squared
# error for a vector of ratings and their corresponding predictors.

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#-----------------------------------------------------------------------------
# Develop the model using the training dataset
#-----------------------------------------------------------------------------

# Establish a baseline - the average rating for all users
mu <- mean(edx$rating)
avg_rmse <- RMSE(edx$rating, mu)

# set up a dataframe to hold the results of this average and modeling RMSEs
rmse_results <- data_frame(method = "Just the average", RMSE = avg_rmse)

head(edx)
# Introduce b_i, to account for general user bias in ratings
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Generate predicted ratings returned by the model
predicted_ratings <- mu + edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# Calculate the RMSE
model_1_rmse <- RMSE(predicted_ratings, edx$rating)

# Add the results to the rmse_results summary
rmse_results <- bind_rows(rmse_results,
                data_frame(method="Movie Effect Model",  
                RMSE = model_1_rmse))

# Improve the model by interoducing b_u, to account for user-specific effects
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Examine ratings returned by the improved model
predicted_ratings <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Add the improved model prediction to the rmse_results summary
model_2_rmse <- RMSE(predicted_ratings, edx$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse))
rmse_results
# 1 Just the average           1.05 
# 2 Movie Effect Model         0.942
# 3 Movie + User Effects Model 0.857

#----------------------------------------------------------------------------- 
# Testing the algorithm with the test data set
#-----------------------------------------------------------------------------

# Establish a baseline - the average rating for all users - and calculate RMSE
test_mu       <- mean(validation$rating)
test_avg_rmse <- RMSE(validation$rating, test_mu)

# set up a dataframe to hold the results of this and other model RMSEs
rmse_results_test <- data_frame(method = "Just the average", RMSE = test_avg_rmse)

# Introduce b_i, to account for general user bias in ratings
movie_avgs <- validation %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - test_mu))

# Examine ratings returned by the revised / fitted model
predicted_ratings <- test_mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# Calculate the RMSE 
mudel_1_rmse      <- RMSE(predicted_ratings, validation$rating)

# Add to the rmse_results_test summary
rmse_results_test <- bind_rows(rmse_results_test,
                     data_frame(method="Movie Effect Model (Test Data)",  
                     RMSE = model_1_rmse))

# Improve the model by interoducing bu, to account for user-specific effects
user_avgs <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - test_mu - b_i))

# Calculate the ratings returned by the improved model
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = test_mu + b_i + b_u) %>%
  pull(pred)

# Calculate the RMSE
model_2_rmse      <- RMSE(predicted_ratings, validation$rating)

# Add the improved model prediction to the rmse_results summary
rmse_results_test <- bind_rows(rmse_results_test,
                          data_frame(method="Movie + User Effects Model (Test Data)",  
                                     RMSE = model_2_rmse))
rmse_results_test

#------------------------------------------------------------------------
# REMOVE THE SECTION BELOW BEFORE SUBMITTING
#------------------------------------------------------------------------

# Summary information for reference
# Dataset has 9000055 rows and 6 columns (dim(edx))
# There are 10677 unique movie IDs
# There are 69878 unique users

# Useful code pieces

# Numbers of different genres
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Most ratings
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# The five most given ratings in order from most to least?
edx %>% group_by(rating) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

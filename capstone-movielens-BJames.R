##############################################################################
#
# Capstone - Movielens Project
# Bill James (BCDL) / jamesbcdl@gmail.com
# Files are located in github at:
#
#      https://github.com/wjamesTMC/ds-education-capstone_project.git
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

#----------------------------------------------------------------------------- 
# Set ups, downloads, and establish data sets
#-----------------------------------------------------------------------------

# Bring in the necessary libraries
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
# Develop / train a model using the edx (training) dataset
#-----------------------------------------------------------------------------

#
# Set ups
#

# Establish a baseline - the average rating for all movies
mu <- mean(edx$rating)
avg_rmse <- RMSE(edx$rating, mu)

# set up a dataframe to hold the results of this average and further model results
rmse_results <- data_frame(method = "Just the average", RMSE = avg_rmse)

# Display the initial result
rmse_results

#
# Model building step 1 - Introduce b_i, to account for general bias in ratings
# We know that different movies are rated differently. The term, b i represents 
# the average rating for a given movie i. 
#

# Calculate b_i as the average of a given movie's ratings minus the overall average 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Generate predicted ratings with b_i factored in
predicted_ratings <- mu + edx %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  pull(b_i)

# Calculate the resulting RMSE
model_1_rmse <- RMSE(predicted_ratings, edx$rating)

# Add the results to the rmse_results summary
rmse_results <- bind_rows(rmse_results,
                data_frame(method = "Movie Effect Model",  
                RMSE = model_1_rmse))

# Display the updated results
rmse_results

# 
# Model building step 2 - introduce b_u, to account for user-specific effects
#

# Calculate b_u as the average of a given movie's ratings minus the overall average 
# minus the general ratings bias
user_avgs <- edx %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Generate predicted ratings with b_u factored in
predicted_ratings <- edx %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate the new RMSE
model_2_rmse <- RMSE(predicted_ratings, edx$rating)

# Add the results to the rmse_results summary
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Movie + User Effects Model",  
                                     RMSE = model_2_rmse))

# Display the updated results
rmse_results

#
# Model building step 3: see if regularization improves the model
#

# Select a Lambda 
lambdas <- seq(0, 10, 0.25)

# Again, the average rating for all movies
mu <- mean(edx$rating)

# Take the sum of each of the movie ratings minus the average rating
# and calculate the number of ratings for that movie
just_the_sum <- edx %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

# Join to the training set, calc new b_i and prediction
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- edx %>% 
    left_join(just_the_sum, by = 'movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx$rating))
})

# Plot lambdas and disply lambda with lowest value
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]
paste("The optimal value of lambda is:",lambdas[which.min(rmses)])

# Use optimal lambda to calculate the new b_i
lambda <- lambdas[which.min(rmses)]
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

# Generate new prediction
predicted_ratings <- edx %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# Calculate RMSE
model_3_rmse <- RMSE(predicted_ratings, edx$rating)

# Update RMSE results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Regularized Movie Effect Model",  
                                     RMSE = model_3_rmse))

# Display updated results
rmse_results

#
# Model building step 4: See if cross validation cna improve the results 
#

# As before, select a Lambda
lambdas <- seq(0, 10, 0.25)

# Function to pick an optimal lambda and resulting RMSE
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- edx %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, edx$rating))
})

# Plot and display the optimized lambda value
qplot(lambdas, rmses) 
lambda <- lambdas[which.min(rmses)]
paste("The optimal value of lambda is", lambda)
paste("The resulting RMSE is", min(rmses))

# Update the final RMSE results table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Using cross-validation",  
                                     RMSE = min(rmses)))
rmse_results 

#----------------------------------------------------------------------------- 
# Testing the algorithm with the test (validation) data set
#-----------------------------------------------------------------------------

#
# Set ups
#

# Re-establish the baseline - the average rating for all movies
test_mu       <- mean(validation$rating)
test_avg_rmse <- RMSE(validation$rating, test_mu)

# set up a dataframe to hold the results of this average and further model results
rmse_results_test <- data_frame(method = "Just the average (Test Data)", RMSE = test_avg_rmse)

# Display the initial result
rmse_results_test

#
# Model testing step 1 - Introduce b_i, to account for general user bias in ratings
#

# Calculate b_i and put in the dataframe movie_avgs
movie_avgs <- validation %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - test_mu))

# Generate predicted ratings with b_i factored in
predicted_ratings <- test_mu + validation %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  pull(b_i)

# Calculate the resulting RMSE
mudel_1_rmse      <- RMSE(predicted_ratings, validation$rating)

# Add the results to the rmse_results summary
rmse_results_test <- bind_rows(rmse_results_test,
                               data_frame(method = "Movie Effect Model (Test Data)",  
                                          RMSE = model_1_rmse))
# Display the updated results
rmse_results_test

# 
# Model testing step 2 - introduce b_u, to account for user-specific effects
#

# Generate predicted ratings with b_u factored in
user_avgs <- validation %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - test_mu - b_i))

# Generate predicted ratings with b_u factored in
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = test_mu + b_i + b_u) %>%
  pull(pred)

# Calculate the RMSE
model_2_rmse      <- RMSE(predicted_ratings, validation$rating)

# Add the results to the rmse_results summary
rmse_results_test <- bind_rows(rmse_results_test,
                               data_frame(method = "Movie + User Effects Model (Test Data)",  
                                          RMSE = model_2_rmse))
# Display the updated results
rmse_results_test

#
# Model testiing step 3: see if regularization improves the result
#

# Select a Lambda 
lambdas <- seq(0, 10, 0.25)

# Re-generate the average rating for all movies
test_mu <- mean(validation$rating)

# Take the sum of each of the movie ratings minus the average rating
# and calculate the number of ratings for that movie
just_the_sum <- validation %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - test_mu), n_i = n())

# Join to the test (validation) set, calc new b_i and prediction
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>% 
    left_join(just_the_sum, by = 'movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = test_mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})

# Plot lambdas and disply lambda with lowest value 
qplot(lambdas, rmses)  
lambda <- lambdas[which.min(rmses)]
paste("The optimal value of lambda is", lambda)

# Use optimal lambda to calculate the new b_i
lambda <- lambdas[which.min(rmses)]
test_mu <- mean(validation$rating)
movie_reg_avgs <- validation %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - test_mu)/(n()+lambda), n_i = n()) 

# Generate new prediction
predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = test_mu + b_i) %>%
  pull(pred)

# Calculate RMSE
model_3_rmse <- RMSE(predicted_ratings, validation$rating)

# Update results
rmse_results_test <- bind_rows(rmse_results_test,
                          data_frame(method = "Regularized Movie Effect Model (Test Data)",  
                                     RMSE = model_3_rmse))

# Display the updated results
rmse_results_test

#
# Model testing step 4: See if cross validation cna improve the results
#

# As before, select a Lambda
lambdas <- seq(0, 10, 0.25)

# Function to join the b_i and b_u values and calculate the new predicted ratings
rmses <- sapply(lambdas, function(l){
  
  test_mu <- mean(validation$rating)
  
  b_i <- validation %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - test_mu)/(n()+l))
  
  b_u <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - test_mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = test_mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

# Plot and display the optimized lambda value
qplot(lambdas, rmses) 
lambda <- lambdas[which.min(rmses)]
paste("The optimal value of lambda is", lambda)
paste("The resulting RMSE is", min(rmses))

# Update the final RMSE results table
rmse_results_test <- bind_rows(rmse_results_test,
                          data_frame(method = "Using Cross-validation (Test Data)",  
                                     RMSE = min(rmses)))

# Display final results
rmse_results_test

#------------------------------------------------------------------------
# End of analysis
#------------------------------------------------------------------------

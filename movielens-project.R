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

# Further model improvements via regularization

# In this video, we're going to introduce the concept of regularization and show
# how it can improve our results even more. This is one of the techniques that
# was used by the winners of the Netflix challenge. All right. So how does it
# work? Note that despite the large movie to movie variation, our improvement in
# residual mean square error when we just included the movie effect was only
# about 5%. So let's see why this happened. Let's see why it wasn't bigger.
# Let's explore where we made mistakes in our first model when we only used
# movies. Here are 10 of the largest mistakes that we made when only using the
# movie effects in our models. Here they are.

test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) 
#                                               title  residual
# 1      Day of the Beast, The (Día de la Bestia, El)  4.500000
# 2                                    Horror Express -4.000000
# 3                                   No Holds Barred  4.000000
# 4  Dear Zachary: A Letter to a Son About His Father -4.000000
# 5                                             Faust -4.000000
# 6                                      Hear My Song -4.000000
# 7                       Confessions of a Shopaholic -4.000000
# 8        Twilight Saga: Breaking Dawn - Part 1, The -4.000000
# 9                                       Taxi Driver -3.806931
# 10                                      Taxi Driver -3.806931

# Note that these all seem to be obscure movies and in our model many of them
# obtained large predictions. So why did this happen? To see what's going on,
# let's look at the top 10 best movies in the top 10 worst movies based on the
# estimates of the movie effect b hat i. So we can see the movie titles, we're
# going to create a database that includes movie ID and titles using this very
# simple code. So here are the best 10 movies according to our estimates.

movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()

movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i) %>% 
  slice(1:10) 

#    title                                                     b_i
#    <chr>                                                   <dbl>
#  1 Lamerica                                                 1.46
#  2 Love & Human Remains                                     1.46
#  3 Enfer, L'                                                1.46
#  4 Picture Bride (Bijo photo)                               1.46
#  5 Red Firecracker, Green Firecracker (Pao Da Shuang Deng)  1.46
#  6 Faces                                                    1.46
#  7 Maya Lin: A Strong Clear Vision                          1.46
#  8 Heavy                                                    1.46
#  9 Gate of Heavenly Peace, The                              1.46
# 10 Death in the Garden (Mort en ce jardin, La)              1.46

# America is number one, Love and Human Remains also number one, Infer L number
# one. Look at the rest of the movies in this table. And here are the top 10
# worst movies.

movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i) %>% 
  slice(1:10) 

#   title                                          b_i
#   <chr>                                        <dbl>
# 1 Santa with Muscles                           -3.04
# 2 B*A*P*S                                      -3.04
# 3 3 Ninjas: High Noon On Mega Mountain         -3.04
# 4 Barney's Great Adventure                     -3.04
# 5 Merry War, A                                 -3.04
# 6 Day of the Beast, The (Día de la Bestia, El) -3.04
# 7 Children of the Corn III                     -3.04
# 8 Whiteboyz                                    -3.04
# 9 Catfish in Black Bean Sauce                  -3.04
# 10 Watcher, The                                 -3.04

# The first one started with Santa with Muscles. Now they all have something in
# common. They're all quite obscure. So let's look at how often they were rated.
# Here's the same table, but now we include the number of ratings they received
# in our training set.

train_set %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) 

#> Joining, by = "movieId"
#> # A tibble: 10 x 3
#>   title                                                     b_i     n
#>   <chr>                                                   <dbl> <int>
#> 1 Lamerica                                                 1.46     1
#> 2 Love & Human Remains                                     1.46     3
#> 3 Enfer, L'                                                1.46     1
#> 4 Picture Bride (Bijo photo)                               1.46     1
#> 5 Red Firecracker, Green Firecracker (Pao Da Shuang Deng)  1.46     3
#> 6 Faces                                                    1.46     1
#> # ... with 4 more rows

train_set %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10)

#> Joining, by = "movieId"
#> # A tibble: 10 x 3
#>   title                                          b_i     n
#>   <chr>                                        <dbl> <int>
#> 1 Santa with Muscles                           -3.04     1
#> 2 B*A*P*S                                      -3.04     1
#> 3 3 Ninjas: High Noon On Mega Mountain         -3.04     1
#> 4 Barney's Great Adventure                     -3.04     1
#> 5 Merry War, A                                 -3.04     1
#> 6 Day of the Beast, The (Día de la Bestia, El) -3.04     1
#> # ... with 4 more rows 

# We can see the same for the bad movies. So the supposed best and worst movies
# were rated by very few users, in most cases just one. These movies were mostly
# obscure ones. This is because with just a few users, we have more uncertainty,
# therefore larger estimates of bi, negative or positive, are more likely when
# fewer users rate the movies. These are basically noisy estimates that we
# should not trust, especially when it comes to prediction. Large errors can
# increase our residual mean squared error, so we would rather be conservative
# when we're not sure. Previously we've learned to compute standard errors and
# construct confidence intervals to account for different levels of uncertainty.
# However, when making predictions we need one number, one prediction, not an
# interval. For this, we introduce the concept of regularization.

# Regularization permits us to penalize large estimates that come from small
# sample sizes. It has commonalities with the Bayesian approaches that shrunk
# predictions. The general idea is to add a penalty for large values of b to the
# sum of squares equations that we minimize. So having many large b's makes it
# harder to minimize the equation that we're trying to minimize. One way to
# think about this is that if we were to fit an effect to every rating, we could
# of course make the sum of squares equation by simply making each b match its
# respective rating y. This would yield an unstable estimate that changes
# drastically with new instances of y. Remember y is a random variable. But by
# penalizing the equation, we optimize to b bigger when the estimate b are far
# from zero. We then shrink the estimates towards zero. Again, this is similar
# to the Bayesian approach we've seen before. So this is what we do.

# To estimate the b's instead of minimizing the residual sum of squares
# as is done by least squares, we now minimize this equation.

#    1N∑u,i(yu,i−μ−bi)2+λ∑ib2

# Note the penalty term. The first term is just the residual sum of squares and
# the second is a penalty that gets larger when many b's are large. Using
# calculus, we can actually show that the values of b that minimized equation
# are given by this formula, where ni is a number of ratings b for movie i.

#    ^bi(λ)=1λ+nini∑u=1(Yu,i−^μ)

# Note that this approach will have our desired effect. When ni is very large
# which will give us a stable estimate, then lambda is effectively ignored
# because ni plus lambda is about equal to ni. However, when ni is small, then
# the estimate of bi is shrunken towards zero. The larger lambda, the more we
# shrink. So let's compute these regularized estimates of vi using lambda equals
# to 3.0. Later we see why we picked this number. So here is the code.

lambda <- 3
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

# To see how the estimates shrink, let's make a plot of the regularized estimate
# versus the least square estimates with the size of the circle telling us how
# large ni was.

data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

#  # A tibble: 10 x 3
#    title                          b_i     n
#    <chr>                         <dbl> <int>
#  1 All About Eve                0.927    26
#  2 Shawshank Redemption, The    0.921   240
#  3 Godfather, The               0.897   153
#  4 Godfather: Part II, The      0.871   100
#  5 Maltese Falcon, The          0.860    47
#  6 Best Years of Our Lives, The 0.859    11
#  # ... with 4 more rows

# You can see that when n is small, the values are shrinking more towards zero.
# All right, so now let's look at our top 10 best movies based on the estimates
# we got when using regularization.

train_set %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10)

# Note that the top five movies are now All
# About Eve, Shawshank Redemption, The Godfather, The Godfather
# II, and the Maltese Falcons.
# This makes much more sense.


# We can also look at the worst movies and the worst five are Battlefield Earth,
# Joe's Apartment, Speed 2, Cross Control, Super Mario Bros, and Police Academy
# 6: City Under Siege. Again, this makes sense.

train_set %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) 

#> # A tibble: 10 x 3
#>   title                                b_i     n
#>   <chr>                              <dbl> <int>
#> 1 Battlefield Earth                  -2.06    14
#> 2 Joe's Apartment                    -1.78     7
#> 3 Speed 2: Cruise Control            -1.69    20
#> 4 Super Mario Bros.                  -1.60    13
#> 5 Police Academy 6: City Under Siege -1.57    10
#> 6 After Earth                        -1.52     4
#> # ... with 4 more rows

# So do we improve our results? We certainly do.

predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = model_3_rmse))
rmse_results

#   method                          RMSE
#   <chr>                          <dbl>
# 1 Just the average               1.05 
# 2 Movie Effect Model             0.986
# 3 Movie + User Effects Model     0.908
# 4 Regularized Movie Effect Model 0.908

# We get the residual mean squared error all the way down to 0.885 from 0.986.
# So this provides a very large improvement. Now note that lambda is a tuning
# parameter. We can use cross-fertilization to choose it. We can use this code
# to do this.

lambdas <- seq(0, 10, 0.25)

mu <- mean(train_set$rating)
just_the_sum <- train_set %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

# And we see why we picked 3.0 as lambda. One important point. Note that we show
# this as an illustration and in practice, we should be using full
# cross-validation just on a training set without using the test it until the
# final assessment.

# We can also use regularization to estimate the user effect. The equation we
# would minimize would be this one now.

#    1N∑u,i(yu,i−μ−bi−bu)2+λ(∑ib2i+∑ub2u)

# It includes the parameters for the user effects as well. The estimates that
# minimizes can be found similarly to what we do previously. Here we again use
# cross-validation to pick lambda.

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses) 

# We see what lambda minimizes our equation. For the full model including movie
# and user effects, the optimal lambda is 3.75. And we can see that we indeed
# improved our residual mean squared error. Now it's 0.881.

lambda <- lambdas[which.min(rmses)]
lambda
#> [1] 3.75
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()
# 
#      |method                                |      RMSE|
#      |:-------------------------------------|---------:|
#      |Just the average                      | 1.0482202|
#      |Movie Effect Model                    | 0.9862839|
#      |Movie + User Effects Model            | 0.9077043|
#      |Regularized Movie Effect Model        | 0.9077043|
#      |Regularized Movie + User Effect Model | 0.8806419|

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

##############################################################################
#
# Capstone - Movielens Project
# B. James (BCDL)
#
# Descriptioin: Create a movie recommendation system using the MovieLens dataset
# and the tools shown throughout the courses. Use the 10M version of the
# MovieLens dataset.
#
# Train a machine learning algorithm using the inputs in one subset to predict
# movie ratings in the validation set. The project will be assessed by peer
# grading.
#
##############################################################################

#----------------------------------------------------------------------------- 
# Section 1: Set ups, downloads, and establish data sets
#-----------------------------------------------------------------------------

# Necessary libraries

library(tidyverse)
library(tidyr)
library(caret)
library(dplyr)
library(data.table)
library(splitstackshape)

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
# Section 2: 
#-----------------------------------------------------------------------------






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

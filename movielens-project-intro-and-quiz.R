#------------------------------------------------------------
#
# Capstone - Movielens Project
#
#------------------------------------------------------------

#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

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

#
# Movielens Quiz
#
g <- unique(edx$genres)
# 1. Number of rows and columns in the edx dataset
dim(edx)
# [1] 9000055       6

# 2. Number of ratings that were zeros and 3s
num_ratings <- edx %>% filter(rating == 0)
nrow(num_ratings) # 0
num_ratings <- edx %>% filter(rating == 3)
nrow(num_ratings) # [1] 2121240

# 3. How many different movies are in the data set
x <- unique(edx$movieId)
length(x) # [1] 10677 Note: unique title does not work (10676)

# 4. Number of unique users
x <- unique(edx$userId)
length(x) # [1] 69878

library(dplyr)
library(tidyr)
library(data.table)
library(splitstackshape)

# 5. How many movie ratings are in each of the following genres in the edx dataset?

df1 <- separate_rows(edx, genres, sep = "\\|", convert = TRUE)

num_genre <- df1 %>% filter(genres == "Drama")
nrow(num_genre) # [1] 3910127

num_genre <- df1 %>% filter(genres == "Comedy")
nrow(num_genre) # [1] 3540930

num_genre <- df1 %>% filter(genres == "Thriller")
nrow(num_genre) # [1] 2325899

num_genre <- df1 %>% filter(genres == "Romance")
nrow(num_genre) # [1] 1712100

edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# 6. Movie with most ratings
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# 7. What are the five most given ratings in order from most to least?
edx %>% group_by(rating) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

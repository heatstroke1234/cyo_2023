# Create edx and final_holdout_test sets

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
                                    repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)

# CYO car rating dataset:
# https://huggingface.co/datasets/florentgbelidji/car-reviews/resolve/main/train_car.csv

options(timeout = 120)

dl <- "train_car.csv"
if(!file.exists(dl))
  download.file("https://huggingface.co/datasets/florentgbelidji/car-reviews/resolve/main/train_car.csv", dl)

dl_frame <- read.csv(dl)
dl_frame <- subset(dl_frame, select = -c(X, Unnamed..0, Author_Name, Review_Title, Review))
dl_frame <- dl_frame %>% 
  separate(Vehicle_Title, c("Model_Year", "Make", "Model"), 
           extra = "drop", fill = "right")
dl_frame$Model_Year = as.numeric(as.character(dl_frame$Model_Year)) 
dl_frame <- dl_frame[order(dl_frame$Model_Year),]
dl_frame$Review_Date = substr(dl_frame$Review_Date, 11, 12)
dl_frame$Review_Date = paste("20", dl_frame$Review_Date, sep = "")
colnames(dl_frame)[1] ="Review_Year"
dl_frame$Review_Year = as.numeric(as.character(dl_frame$Review_Year)) 

# Final hold-out test set will be 10% of the given data
set.seed(1, sample.kind="Rounding") # using R 3.6 or later
test_index <- createDataPartition(y = dl_frame$Rating, times = 1, 
                                  p = 0.1, list = FALSE)
edx <- dl_frame[-test_index,]
final_holdout_test <- dl_frame[test_index,]

# Further division of edx into training and testing sets
set.seed(1, sample.kind = "Rounding") # using R 3.6 or later
test_index <- createDataPartition(y = edx$Rating, times = 1, 
                                  p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

### Starting data exploration

# Statistical summary of the dataset edx
summary(edx)

# Graph top car models
edx %>%
  group_by(Model) %>%
  summarize(count = n()) %>%
  top_n(10, count) %>%
  arrange(-count) %>%
  ggplot(aes(count, reorder(Model, count))) +
  geom_bar(color = "gray", fill = "firebrick", stat = "identity") +
  labs(x = "Count", y = "Car Models", caption = "Source: given dataset") +
  ggtitle("Most Popular Car Models")
  
# Graph number of ratings per rating
edx %>%
  ggplot(aes(Rating)) +
  geom_bar(color = "gray", fill = "firebrick") +
  labs(x = "Ratings", y = "Frequency", caption = "Source: given dataset") +
  scale_x_continuous(breaks = seq(0, 5, by = 1)) +
  ggtitle("Rating Count Per Rating")

# Graph top car models
edx %>%
  group_by(Model_Year) %>%
  summarize(count = n()) %>%
  top_n(10, count) %>%
  arrange(-count) %>%
  ggplot(aes(count, reorder(Model_Year, count))) +
  geom_bar(color = "gray", fill = "firebrick", stat = "identity") +
  labs(x = "Count", y = "Car Model Years", caption = "Source: given dataset") +
  ggtitle("Top Car Model Year Ratings")

# Graph number of ratings versus car models
edx %>% 
  group_by(Model) %>%
  summarize(count = n()) %>%
  ggplot(aes(count)) +
  geom_histogram(color = "gray", fill = "firebrick", bins = 50) +
  labs(x = "Ratings", y = "Car Models", caption = "Source: given dataset") +
  ggtitle("Number of Ratings Versus Car Models")

### Starting data analysis

# Function to calculate RMSE
rmse <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Mean of all ratings
mean_rating <- mean(train_set$Rating)
mean_rating

# RMSE calculated with just the mean
mean_rmse <- rmse(test_set$Rating, mean_rating)
mean_rmse

# Add car model bias to calculation
bi <- train_set %>%
  group_by(Model) %>%
  summarize(b_i = mean(Rating - mean_rating))
predicted_ratings <- mean_rating + test_set %>%
  left_join(bi, by = "Model") %>%
  pull(b_i)

# RMSE calculated with mean and car model bias
car_model_bias_rmse <- rmse(predicted_ratings, test_set$Rating)
car_model_bias_rmse

# Add car make bias to calculation
bu <- train_set %>%
  left_join(bi, by = "Model") %>%
  group_by(Make) %>%
  summarize(b_u = mean(Rating - mean_rating - b_i))
predicted_ratings <- test_set %>%
  left_join(bi, by = "Model") %>%
  left_join(bu, by = "Make") %>%
  mutate(pred = mean_rating + b_i + b_u) %>%
  pull(pred)

# RMSE calculated with mean, car model, and car make bias
car_make_bias_rmse <- rmse(predicted_ratings, test_set$Rating)
car_make_bias_rmse

# Add review year bias to calculation
bt <- train_set %>%
  left_join(bi, by = "Model") %>%
  left_join(bu, by = "Make") %>%
  group_by(Review_Year) %>%
  summarize(b_t = mean(Rating - mean_rating - b_i - b_u))
predicted_ratings <- test_set %>%
  left_join(bi, by = "Model") %>%
  left_join(bu, by = "Make") %>%
  left_join(bt, by = "Review_Year") %>%
  mutate(pred = mean_rating + b_i + b_u + b_t) %>%
  pull(pred)

# RMSE calculated with mean, car model, car make, and review year bias
review_year_bias_rmse <- rmse(predicted_ratings, test_set$Rating)
review_year_bias_rmse

# Applying data regularization
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(x){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mean_rating)/(n() + x)) # adding movie bias
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - 
                          mean_rating)/(n() + x)) # adding user bias
  b_t <- train_set %>%
    mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(date) %>%
    summarize(b_t = mean(rating - b_i - b_u - 
                           mean_rating)/(n() + x)) # adding time bias
  predicted_ratings <- test_set %>%
    mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = "date") %>%
    mutate(pred = mean_rating + b_i + b_u + b_t) %>%
    pull(pred)
  return(rmse(predicted_ratings, test_set$rating))
})

# Plotting lambdas versus RMSEs
qplot(lambdas, rmses, color = I("red"))

# Finding which lambda has the lowest RMSE
lambda <- lambdas[which.min(rmses)]
lambda

# Selecting the lambda with the lowest RMSE
regularized_rmse <- min(rmses)
regularized_rmse

# Applying matrix factorization using the recosystem package
if(!require(recosystem)) install.packages(
  "recosystem", repos = "http://cran.us.r-project.org")
library(recosystem)
set.seed(1, sample.kind = "Rounding") # using R 3.6 or later
reco_train <- with(train_set, data_memory(user_index = userId, 
                                          item_index = movieId, 
                                          rating = rating))
reco_test <- with(test_set, data_memory(user_index = userId, 
                                        item_index = movieId, rating = rating))
reco <- Reco()

reco_para <- reco$tune(reco_train, opts = list(dim = c(20, 30), 
                                               costp_l2 = c(0.01, 0.1),
                                               costq_l2 = c(0.01, 0.1), 
                                               lrate = c(0.01, 0.1),
                                               nthread = 4, niter = 10))

reco$train(reco_train, opts = c(reco_para$min, nthread = 4, niter = 30))
reco_first <- reco$predict(reco_test, out_memory())

# RMSE calculated with matrix factorization
factorization_rmse <- RMSE(reco_first, test_set$rating)
factorization_rmse

# Using matrix factorization on final holdout test
set.seed(1, sample.kind = "Rounding") # using R 3.6 or later
reco_edx <- with(edx, data_memory(user_index = userId, item_index = movieId, 
                                  rating = rating))
reco_final_holdout <- with(final_holdout_test, data_memory(user_index = userId, 
                                                           item_index = movieId, 
                                                           rating = rating))
reco <- Reco()

reco_para <- reco$tune(reco_edx, opts = list(dim = c(20, 30), 
                                             costp_l2 = c(0.01, 0.1),
                                             costq_l2 = c(0.01, 0.1), 
                                             lrate = c(0.01, 0.1),
                                             nthread = 4, niter = 10))

reco$train(reco_edx, opts = c(reco_para$min, nthread = 4, niter = 30))
reco_final <- reco$predict(reco_final_holdout, out_memory())

# Generating final RMSE
final_rmse <- RMSE(reco_final, final_holdout_test$rating)
final_rmse

### Final results

# Table made using the reactable package
if(!require(reactable)) install.packages("reactable", 
                                         repos = "http://cran.us.r-project.org")
library(reactable)
if(!require(webshot2)) install.packages("webshot2", 
                                         repos = "http://cran.us.r-project.org")
library(webshot2)
if(!require(htmlwidgets)) install.packages("htmlwidgets", 
                                        repos = "http://cran.us.r-project.org")
library(htmlwidgets)
Methods <- c("Just the mean", "Mean and car model bias", 
             "Mean, car model, and car make bias", "Mean, car model, 
             car make, and model year bias", 
             "Regularized movie, user, and time effects",
             "Matrix factorization using recosystem", 
             "Final holdout test 
             (generated using matrix factorization)") # first column
RMSE <- c(round(mean_rmse, 7), round(car_model_bias_rmse, 7), 
          round(car_make_bias_rmse, 7), round(review_year_bias_rmse, 7), 
          round(regularized_rmse, 7), round(factorization_rmse, 7), 
          round(final_rmse, 7)) # second column
final_results <- data.frame(Methods, RMSE)
table <- reactable(final_results,
  highlight = TRUE,
  bordered = TRUE,
  theme = reactableTheme(
    borderColor = "#dfe2e5",
    highlightColor = "#f0f5f9",
    cellPadding = "8px 12px",
    style = list(fontFamily = "-apple-system, BlinkMacSystemFont, 
                 Segoe UI, Helvetica, Arial, sans-serif"),
    )
  )
saveWidget(widget = table, file = "table_html.html", selfcontained = TRUE)
webshot(url = "table_html.html", file = "final_table.png", delay = 0.1, 
        vwidth = 1245)
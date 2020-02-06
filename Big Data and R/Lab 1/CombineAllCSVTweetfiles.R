path <- "C:/DIC/Labs/Lab1/Part3/Tweets/"
setwd(path)
all_files <- list.files(path, pattern=NULL, all.files=FALSE, full.names=FALSE)
data_frame <- data.frame()

for (file in all_files) {
  temp <- read.csv(file)
  data_frame <-  rbind(data_frame, temp)
}
write.csv(data_frame, paste("C:/DIC/Labs/Lab1/Part3/", "AllTweets",".csv"))
library(jsonlite)
library("maps")
library(revgeo)
library(twitteR)
library(rtweet)
library("ggmap")
library("maptools")
library(stringr)

register_google(key = 'AIzaSyCg_VUNB8jbTN5tQjdUCEBig-QwsLlhx2s')

token_app <- create_token("Kprabhak's IR Proj-1", "3nar7eM7oWjrgeUQH8RvhlTDO", "RnWg5gITK5Mwo3FutcoX8d0ZLVsLp8PsrL0md4B80FDkcMnLZ2",
                          "1035703525324541952-WxntgPGzowU90aUr3poOqGLwhzDr6Q", "i0jquCYVa8Y5vUvEJl2DFIgYyVkV48EGYm8laEtMlcPoS")

register_google(key = 'AIzaSyCg_VUNB8jbTN5tQjdUCEBig-QwsLlhx2s') 


setwd('C:/Semester1/IR/projects/PythonCrawler/new-python-crawler/tweets_DIC/')
file <- "H1N1__en_1000_2019-02-20_2019-03-02_lang.json"

tweets <- fromJSON(file, flatten = TRUE)

tweets_df <- as.data.frame(tweets)

df <- tweets_df[, c('id', 'text','queryMetadata.query', 'created_at', 'user.screen_name')]
df["longitude"] <- NA
df["latitude"] <- NA
df["state"] <- NA
df["city"] <- NA

x <- vapply(df$text, length, 1L)
df <- df[rep(rownames(df), x), ]  
df$text <- unlist(df$text, use.names = FALSE)

for(i in 1:nrow(df)) 
{
  row_data <-df[i:i, ]
  if(is.na(row_data$latitude))
  {
    users <- c(row_data$user.screen_name)
    usr_df <- lookup_users(users, token = token_app)
    usr_df$location
    
    # reverseGeocoding
    geo_output <- tryCatch({
      loc <- geocode(c( usr_df$location))
      if(str_detect(revgeocode(as.numeric(loc)), "USA")){
        df[i:i, ]["latitude"] <- loc$lat
        df[i:i, ]["longitude"] <- loc$lon
        df[i:i, ]["state"] <- revgeo(longitude=loc$lon, latitude=loc$lat, output = 'hash', item = 'state', provider = 'google', API = 'AIzaSyCg_VUNB8jbTN5tQjdUCEBig-QwsLlhx2s')
        df[i:i, ]["city"] <- revgeo(longitude=loc$lon, latitude=loc$lat, output = 'hash', item = 'city', provider = 'google', API = 'AIzaSyCg_VUNB8jbTN5tQjdUCEBig-QwsLlhx2s')
      }
      else{
        print(revgeocode(as.numeric(loc)))
      }
    }, warning = function(w) {
      print(usr_df$location)
      print(w)
    }, error = function(e) {
      print(usr_df$location)
      print(e)
    }, finally = {
      
    })
    geo_output
  }
}
df <- df[which(!is.na(df$latitude)), ]
write.csv(df, paste("C:/DIC/Labs/Lab1/Part3/", file,".csv"))

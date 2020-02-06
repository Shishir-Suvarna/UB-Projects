library(twitteR)
library(rtweet)
library("ggmap")
library("maptools")
library("maps")

setup_twitter_oauth("3nar7eM7oWjrgeUQH8RvhlTDO", "RnWg5gITK5Mwo3FutcoX8d0ZLVsLp8PsrL0md4B80FDkcMnLZ2",
                    "1035703525324541952-WxntgPGzowU90aUr3poOqGLwhzDr6Q", "i0jquCYVa8Y5vUvEJl2DFIgYyVkV48EGYm8laEtMlcPoS")
token_app <- create_token("Kprabhak's IR Proj-1", "3nar7eM7oWjrgeUQH8RvhlTDO", "RnWg5gITK5Mwo3FutcoX8d0ZLVsLp8PsrL0md4B80FDkcMnLZ2",
             "1035703525324541952-WxntgPGzowU90aUr3poOqGLwhzDr6Q", "i0jquCYVa8Y5vUvEJl2DFIgYyVkV48EGYm8laEtMlcPoS")

register_google(key = 'AIzaSyCg_VUNB8jbTN5tQjdUCEBig-QwsLlhx2s') 


searchTerms <-c('#flu','#influenza','#antiketombe','#ketombe','#cold','#antibakteria',
                 '#antibakteri','#antiserangga','#fluview','Influenza+A','Influenza+B','#H1N1','#H3N2',
                 '#H3','#H1','#ILI','influenza-associated','#pneumonia','#P&I','#CDC','flu+virus',
                 'public+health+laboratories','clinical+laboratories','pdm09','#HHS','WHO+flu','NREVSS+flu',
                 '#flu_by_age_virus','#flushot','#flutest','#fluvirus','#influenzavirus','Yamagata+lineage',
                 'Victoria+lineage','flu+positive','flu+negative','#fluactivity','influenza+activity',
                 '#fluvaccine','flu+vaccination','Phylogenetic+analysis','neuraminidase','oseltamivir',
                 'zanamivir','peramivir','baloxavir','adamantanes','flu+Virus+Susceptibility','#ILINet',
                 '#ILIactivity','ILI+activity','FluSurv','#FluSurv','#FluSurv-NET','Influenza+Hospitalization',
                 'HHS+region','#PedFluDeath','#fluwatch','#flunews','#FluNET','#FluSeason','FLu+Activity',
                 'Flu+Surveillance','Prevent+Flu','FluSight','FluForecasting','Flu+Forecasting','#FluVax',
                 '#FluNews','FluNews','Flu+mails','#flufightercare','#fluwatch2019')
searchTerms <- unique(searchTerms)


library("ggmap")
library("maptools")
library("maps")
library(stringr)
register_google(key = 'AIzaSyCg_VUNB8jbTN5tQjdUCEBig-QwsLlhx2s')

for (topic in searchTerms) {
  # for (code in geocodes) {
    Sys.sleep(1)
   output <- tryCatch({
      result <- searchTwitter(topic, retryOnRateLimit = 10, since='2019-01-01', until='2019-03-01') # ) # , n=1000) #, lang=NULL,geocode= code, retryOnRateLimit = 10)
      df <- twListToDF(result)
      df <- df[which(!df$isRetweet & !df$retweeted), ]
      df$topic <- topic
      print(paste("number of tweets obtained is..", nrow(df)))
      for(i in 1:nrow(df)) 
      {
        row_data <-df[i:i, ]
        if(is.na(row_data$latitude))
        {
          users <- c(row_data$screenName)
          
          ## get users data
          usr_df <- lookup_users(users, token = token_app)
          
          ## view users data
          usr_df$location
          
         
          # reverseGeocoding
          geo_output <- tryCatch({
            loc <- geocode(c( usr_df$location))
          if(str_detect(revgeocode(as.numeric(loc)), "USA")){
            df[i:i, ]$longitude <- loc$lon
            df[i:i, ]$latitude <- loc$lat
          }
            else{
              print(revgeocode(as.numeric(loc)))
            }
          }, warning = function(w) {
            print(w)
          }, error = function(e) {
            print(e)
          }, finally = {
            
          })
          geo_output
        }
      }
      df <- df[which(!is.na(df$latitude)), ]
      write.table(df, paste("tweets-until", ".csv"),sep = ",", append = TRUE, col.names = FALSE)
      print(paste("number of tweets added to file is..", nrow(df)))
      # if(substr(topic, start=1, stop=1) == '#'){
      #   write.csv(df, paste("H_", substr(topic, start=2, stop=nchar(topic)), ".csv"))
      # }
      # else{
      #   write.csv(df, paste(topic, ".csv")) # append = TRUE to append
      # }
    }, warning = function(w) {
       print(w)
    }, error = function(e) {
      print(e)
    }, finally = {
     
    })
   output
   print(paste('completed search for topic: ', topic))
  # }
}


#### Testing Code  ####

# visited <- c("Bengaluru, India","SFO", "New York", "Buffalo", "Dallas, TX")
# geocode(visited) # gets the lat and long of the user location
# ## view tweet data for these users via tweets_data()
# #tweets_data(usr_df)
# cities <- us.cities
# geocodes <- paste(cities$lat, ",", cities$long, ",", "250mi")
# result <- searchTwitter(searchTerms[1], n=1000, lang=NULL, retryOnRateLimit = 10)
# df <- twListToDF(result)
# searchTerms[1:3]

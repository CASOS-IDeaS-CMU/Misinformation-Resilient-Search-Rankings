library("RAhrefs")
library("httr")
library("jsonlite")

# -------------- AUTH ---------------
api_key <- "<INSERT KEY>"
RAhrefs::rah_auth(api_key)

# -------------- CONFIG ---------------

urls <- read.csv("C:/dev/seo_backlink_network/data/traffic/attributes_unfiltered.csv")[,1,drop=TRUE]
file_prefix = 'C:/dev/seo_backlink_network/data/traffic/output_2'

get_ahref_rank <- function(target) {
  report = "ahrefs_rank"
  mode = "subdomains"
  limit = 1
  response <- GET(paste0("https://apiv2.ahrefs.com/", "?token=", 
                         api_key, "&from=", report, "&target=", target, "&mode=", 
                         mode, "&limit=", limit, "&output=json"))
  stop_for_status(response)
  content <- content(response, type = "text", encoding = "UTF-8")
  result <- fromJSON(content, simplifyVector = FALSE)
  return(result)
}

# Node list
traffic_df <- data.frame(matrix(ncol = 3, nrow = 0))
colnames(traffic_df) <- c('url', 'url_2', 'rank')

for (url in urls[3650:]) {
  # downloading data -------------------
  print(url)
  
  traffic_data <- 
  traffic_data <- try(get_ahref_rank(
    target = url
  ))
  if(inherits(traffic_data, "try-error"))
  {
    #error handling code, maybe just skip this iteration using
    print(paste("Failed rank: ",url))
    next
  }
  traffic_data <- do.call(rbind.data.frame, traffic_data[[1]])
  traffic_data <- cbind(url = url, traffic_data)
  traffic_df[nrow(traffic_df) + 1,] = c(traffic_data)
  
  write.csv(traffic_df, paste(file_prefix, '_rank.csv'), row.names = FALSE)
}

write.csv(traffic_df, paste(file_prefix, '_rank.csv'), row.names = FALSE)
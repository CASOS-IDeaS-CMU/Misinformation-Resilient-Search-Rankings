library("RAhrefs")
library("httr")
library("jsonlite")

# -------------- AUTH ---------------
api_key <- "<INSERT KEY>"
RAhrefs::rah_auth(api_key)

# -------------- CONFIG ---------------

urls <- read.csv("C:/dev/seo_backlink_network/data/traffic/attributes_unfiltered.csv")[,1,drop=TRUE]
file_prefix = 'C:/dev/seo_backlink_network/data/traffic/output_2'

get_ahref_position_metrics <- function(target) {
  report = "positions_metrics"
  mode = "subdomains"
  limit = 2
  response <- GET(paste0("https://apiv2.ahrefs.com/", "?token=", 
                         api_key, "&from=", report, "&target=", target, "&mode=", 
                         mode, "&limit=", limit, "&output=json"))
  stop_for_status(response)
  content <- content(response, type = "text", encoding = "UTF-8")
  result <- fromJSON(content, simplifyVector = FALSE)
  return(result)
}

# Node list
traffic_df <- data.frame(matrix(ncol = 10, nrow = 0))
colnames(traffic_df) <- c('url', 'positions', 'positions_top10', 'positions_top3', 'traffic', 'traffic_top10', 'traffic_top3', 'cost', 'cost_top10', 'cost_top3')

for (url in urls) {
  # downloading data -------------------
  print(url)
  
  traffic_data <- try(get_ahref_position_metrics(
    target = url
  ))
  if(inherits(traffic_data, "try-error"))
    {
      #error handling code, maybe just skip this iteration using
      print(paste("Failed traffic: ",url))
      next
    }
  traffic_data <- do.call(rbind.data.frame, traffic_data)
  traffic_data <- cbind(url = url, traffic_data)
  traffic_df[nrow(traffic_df) + 1,] = c(traffic_data)
  
  write.csv(traffic_df, paste(file_prefix, '_traffic.csv'), row.names = FALSE)
}

write.csv(traffic_df, paste(file_prefix, '_traffic.csv'), row.names = FALSE)
library("RAhrefs")

# -------------- AUTH ---------------
api_key <- "<INSERT KEY>"
RAhrefs::rah_auth(api_key)

# -------------- CONFIG ---------------
urls <- read.csv("remaining_urls.csv")[,1,drop=TRUE]
file_prefix = "politicalnews/discovered_2"

# Node list
url_df <- data.frame(matrix(ncol = 24, nrow = 0))
colnames(url_df) <- c('url', 'backlinks', 'refpages', 'pages', 'valid_pages',   'text', 'image', 'nofollow',  'ugc', 'sponsored', 'dofollow', 'redirect', 'canonical', 'gov', 'edu', 'rss', 'alternate', 'html_pages', 'links_internal', 'links_external', 'refdomains', 'refclass_c', 'refips', 'linked_root_domains')
count = 0

for (url in urls) {
  print(paste(count, ": ", url))
  count = count + 1
  
  url_data <- try(RAhrefs::rah_metrics_extended(
    target = url,
    mode = "subdomains",
  ))
  if(inherits(url_data, "try-error"))
  {
    #error handling code, maybe just skip this iteration using
    print(paste("Failed outlink: ",url))
    next
  }
  url_data <- cbind(url = url, url_data)
  url_df[nrow(url_df) + 1,] = c(url_data)
  write.csv(url_df, paste(file_prefix, '_seo_attributes.csv', sep = ""), row.names = FALSE)
}

write.csv(url_df, paste(file_prefix, '_seo_attributes.csv', sep = ""), row.names = FALSE)
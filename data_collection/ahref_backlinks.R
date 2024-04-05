library("RAhrefs")

# -------------- AUTH ---------------
api_key <- "<INSERT KEY>"
RAhrefs::rah_auth(api_key)

# -------------- CONFIG ---------------
urls <- read.csv("traffic/backlink_edges_20k_bl_2k_rp.csv")[,1:2,drop=TRUE]
backlink_limit = 20
file_prefix = 'traffic/output_backlinks_unagg'

# Edge list
link_df <- data.frame(matrix(ncol = 5, nrow = 0))
colnames(link_df) <- c('domain_from',   'domain_to',  'links', 'unique_pages', 'domain_to_rating')

count = 0

for (i in 1:nrow(urls)) {
  domain_from = urls[[1]][i]
  domain_to = urls[[2]][i]
  print(paste(count, ": ", domain_from, ' -> ', domain_to))
  count = count + 1
  
  # downloading data -------------------
  url_backlinks <- try(RAhrefs::rah_refdomains(
    target = url,
    mode = "subdomains",
    limit = backlink_limit,
    order_by = "backlinks:desc"
  ))
  if(inherits(url_backlinks, "try-error"))
  {
    #error handling code, maybe just skip this iteration using
    print(paste("Failed backlinks: ",url))
    next
  }
  url_backlinks = subset(url_backlinks, select = -c(first_seen, last_visited))
  colnames(url_backlinks) <- c('domain_from', 'links', 'unique_pages', 'domain_to_rating')
  
  for (i in 1:backlink_limit) {
    # swap to & from
    backlink_df = c(domain_to=url, url_backlinks[i,])
    temp = backlink_df[1]
    backlink_df[1] = backlink_df[2]
    backlink_df[2] = temp
    
    link_df[nrow(link_df) + 1,] = backlink_df
  }

  write.csv(link_df, paste(file_prefix, '_edges.csv'), row.names = FALSE)
}

# write.csv(link_df, paste(file_prefix, '_edges.csv'), row.names = FALSE)

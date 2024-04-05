library("RAhrefs")

# -------------- AUTH ---------------
api_key <- "<INSERT KEY>"
RAhrefs::rah_auth(api_key)

# -------------- CONFIG ---------------
urls <- read.csv("traffic/backlink_edges_20k_bl_2k_rp_reduced.csv")[,1:2,drop=TRUE]
backlink_limit = 50
file_prefix = 'traffic/output_backlinks_unagg'

# Edge list
link_df <- data.frame(matrix(ncol = 29, nrow = 0))
colnames(link_df) <- c('domain_from',   'domain_to',  'links', 'unique_pages', 'domain_to_rating')

count = 0


for (i in 611:nrow(urls)) {
  domain_from = urls[[1]][i]
  domain_to = urls[[2]][i]
  print(paste(count, ": ", domain_from, ' -> ', domain_to))
  count = count + 1
  
  matching_url <- RAhrefs::rah_condition(
    column_name = "url_from",
    operator = "SUBSTRING",
    value = domain_from
  )
  
  # downloading data -------------------
  url_backlinks <- try(RAhrefs::rah_backlinks(
    target = domain_to,
    token = api_key,
    mode = "subdomains",
    limit = backlink_limit,
    where = RAhrefs::rah_condition_set(matching_url)
  ))
  if(inherits(url_backlinks, "try-error"))
  {
    #error handling code, maybe just skip this iteration using``
    print(paste("Failed backlinks: ",url))
    next
  }

  write.table(url_backlinks, paste(file_prefix, '_edges.csv'), row.names = FALSE, col.names=FALSE,append=TRUE)
}
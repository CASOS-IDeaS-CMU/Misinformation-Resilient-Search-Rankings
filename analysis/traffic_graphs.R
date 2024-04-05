library(tidyverse)

## Code to compare ahrefs traffic with that of SimilarWeb. 
# Also compares similarweb rank with CommonCrawl Rank.


sim <- read_csv('../data/simweb.csv')
ccout <- read.csv('../cc-webgraph/ranking/output/ranks/exp-baseline.label_only.out') %>% select(domain = url, pagerank_rank, harmonic_rank)
cc <- read_csv('../data/traffic.csv')
sim <- sim %>% select(domain, sim_jan_est = `estimatedMonthlyVisits/2024-01-01`, sim_bounce = bounceRate, 
                      sim_rank = globalRank, sim_time = timeOnSite, sim_ppv = pagesPerVisit)
cc <- cc %>% select(domain = url, ahrefs_traffic = traffic, pr = `cc-orig-pr`, hr = `cc-orig-hr`, label)
df <- sim %>% left_join(cc) %>% drop_na()
df <- df %>% left_join(ccout)
df$reliability <- as.factor(ifelse(df$label > 4, 'reliable', 'unreliable'))

# Check traffic correlation
cor(df$sim_jan_est, df$ahrefs_traffic)
cor(log(df$sim_jan_est+1), log(df$ahrefs_traffic +1))

summary(lm(log(df$ahrefs_traffic +1)~log(df$sim_jan_est+1)))

sum((df$sim_jan_est/31) > df$ahrefs_traffic)/ dim(df)[1]
summary(df)
cor(df$sim_jan_est,df$ahrefs_traffic,method="spearman")

df <- df %>% drop_na()
cor(df$sim_jan_est,df$pagerank_rank,method="spearman")

cor(log(df$sim_jan_est),log(df$pr), method = 'spearman')

# Compare Ahrefs and Simweb Traffic
ggplot(df, aes(x=log10(sim_jan_est+1), y=log10(ahrefs_traffic+1), color = reliability)) + geom_point() +  
  geom_smooth(aes(group=1),color='#A9A9A9')+
  scale_color_manual(values = c("#4169E1","tomato3")) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line=element_line(color='black'))+
  theme(text = element_text(size = 25),
        axis.title.y = element_text(size=25),
        axis.title.x = element_text(size=25)) +
  xlab('Log SimilarWeb Traffic Predictions')+
  ylab('Log Ahrefs Traffic Predictions')
ggsave('../fig/traffic_comparison.png', dpi = 300, height = 9, width = 11, units = 'in')

## Compare similar pagerank rank with commoncrawl pagerank rank
df <- df %>% drop_na()
cor(df$sim_rank,df$pagerank_rank,method="spearman")
cor(log(df$sim_rank+1), log(df$pagerank_rank +1))

summary(lm(log(df$sim_rank +1)~log(df$pagerank_rank+1)))

df <- df %>% drop_na()
cor(log10(df$ahrefs_traffic+1), log10(df$pr))
cor(log10(df$sim_jan_est),log10(df$pr))

summary(lm(log(df$sim_jan_est +1)~log(df$pr))) #0.59
summary(lm(log(df$ahrefs_traffic +1)~log(df$pr))) #0.62


# Compare PageRank with Simweb Rank
ggplot(df, aes(x=log10(sim_jan_est), y=log10(pr), color = reliability)) + geom_point() +  
  geom_smooth(aes(group=1),color='#A9A9A9')+
  scale_color_manual(values = c("#4169E1","tomato3")) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line=element_line(color='black'))+
  theme(text = element_text(size = 25),
        axis.title.y = element_text(size=25),
        axis.title.x = element_text(size=25)) +
  xlab('Log SimilarWeb Traffic estimate')+
  ylab('Log Commoncrawl PageRank')
ggsave('../fig/log_pagerank_comparison.png', dpi = 300, height = 9, width = 11, units = 'in')

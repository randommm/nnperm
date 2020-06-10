library(RPostgreSQL)
library(tidyverse)

drv <- dbDriver("PostgreSQL")

pgdb <- Sys.getenv('pgdb')
pguser <- Sys.getenv('pguser')
pgpass <- Sys.getenv('pgpass')
pghost <- Sys.getenv('pghost')
pgport <- Sys.getenv('pgport')

if (!exists("con")) {
  con <- dbConnect(drv, user=pguser, password=pgpass,
    dbname=pgdb, host=pghost, port=pgport)

  rs <- dbSendQuery(con, "select * from result where complexity = 1 and method != 'remove' and distribution!=4 and distribution!=5")
  df <- fetch(rs, n = -1)
}

for (method in c("cpi", "scpi")) {
power <- df %>%
  filter(complexity == 1, method != "remove",
         distribution!=4, distribution!=5,
         ) %>%
  filter(method =="cpi" | retrain_permutations==1) %>%
  group_by(db_size,betat,estimator,distribution,method) %>%
  summarise(power=mean(pvalue<0.05)) %>%
  ungroup()

power$estimator[power$estimator=="rf"] <- "RF"
power$estimator[power$estimator=="ann"] <- "ANN"
power$estimator[power$estimator=="linear"] <- "LINEAR"
power$method[power$method=="permutation"] <- "COINP"
power$method[power$method=="shuffle_once"] <- "SCPI"
power$method[power$method=="cpi"] <- "CPI"
power <- power[power$method==toupper(method)|power$method=="COINP", ]
power$distribution <- paste("Distribution",power$distribution+1)
colnames(power)[colnames(power)=="estimator"] <- "Test"

ggplot(power %>% filter(db_size==1000))+
  geom_line(aes(x=betat,y=power,color=Test,linetype=method),size=1.10)+
  geom_point(aes(x=betat,y=power,color=Test,shape=Test),size=1.8)+
  facet_wrap(.~distribution,nrow=3)+
  theme_minimal(base_size = 14)+
  scale_linetype_discrete(name="Learning method")+
  xlab(expression(beta[S]))+
  ylab("Power")
ggsave(filename = paste0("plots/power_1000_",method,".pdf"),width = 8,height = 7)

ggplot(power %>% filter(db_size==10000))+
  geom_line(aes(x=betat,y=power,color=Test,linetype=method),size=1.0)+
  geom_point(aes(x=betat,y=power,color=Test,shape=Test),size=1.8)+
  facet_wrap(.~distribution,nrow=3)+
  theme_minimal(base_size = 14)+
  scale_linetype_discrete(name="Learning method")+
  xlab(expression(beta[S]))+
  ylab("Power")
ggsave(filename = paste0("plots/power_10000_",method,".pdf"),width = 8,height = 7)
}

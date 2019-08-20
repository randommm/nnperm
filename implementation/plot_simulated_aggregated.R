library(RPostgreSQL)
library(tidyverse)

drv <- dbDriver("PostgreSQL")

con <- dbConnect(drv,user="",
                 password="",
                 dbname="", host="", port="")

rs <- dbSendQuery(con, "select * from result")
df <- fetch(rs, n = -1)

power <- df %>%
  filter(complexity == 1,retrain_permutations==1,method!="remove",
         distribution!=4) %>%
  group_by(db_size,betat,estimator,distribution,method) %>%
  summarise(power=mean(pvalue<0.05)) %>%
  ungroup()

power$estimator[power$estimator=="rf"] <- "RF"
power$estimator[power$estimator=="ann"] <- "ANN"
power$estimator[power$estimator=="linear"] <- "LINEAR"
power$method[power$method=="permutation"] <- "COINP"
power$method[power$method=="shuffle_once"] <- "CPI"
power$distribution <- power$method+1
colnames(power)[colnames(power)=="estimator"] <- "Test"

ggplot(power %>% filter(db_size==1000))+
  geom_line(aes(x=betat,y=power,color=Test,linetype=method),size=1.10)+
  geom_point(aes(x=betat,y=power,color=Test,shape=Test),size=1.8)+
  facet_wrap(.~distribution,nrow=2)+
  theme_minimal(base_size = 14)+
  scale_linetype_discrete(name="Learning method")+
  xlab(expression(beta[S]))+
  ylab("Power")
ggsave(filename = "power_1000.png",width = 8,height = 5)

ggplot(power %>% filter(db_size==10000))+
  geom_line(aes(x=betat,y=power,color=Test,linetype=method),size=1.0)+
  geom_point(aes(x=betat,y=power,color=Test,shape=Test),size=1.8)+
  facet_wrap(.~distribution,nrow=2)+
  theme_minimal(base_size = 14)+
  scale_linetype_discrete(name="Learning method")+
  xlab(expression(beta[S]))+
  ylab("Power")
ggsave(filename = "power_10000.png",width = 8,height = 5)

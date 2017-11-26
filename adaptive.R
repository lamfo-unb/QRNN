rm(list=ls()) # limpa o workspace
library(ggplot2)
# Define o percentil mirado para o VaR
tau <- 0.05

# Define o ativo para calcular o VaR ('GM' ou 'IBM' ou 'SP500')
ativo <- "Adjusted.Close" 

# Carrega os dados utilizados no artigo do Engels
stocks <- read.table("./SP500.csv", sep = ',', header = T)
stocks <- stocks[seq(dim(stocks)[1],1),] # reverte as linhas
stocks[,2:7] <- log(stocks[,2:7])
stocks[2:nrow(stocks),2:7] <- (stocks[2:nrow(stocks),2:7] - stocks[1:nrow(stocks)-1,2:7]) * 100
stocks = stocks[2:nrow(stocks), ]
stocks$Date = as.Date(stocks$Date)

# Separa as observações me treino e teste, assim como no artigo do Engels
train<-stocks[1:(nrow(stocks)-500),]
valid<-stocks[(nrow(stocks)-500):nrow(stocks),]

# Função custo da regressão quantílica
loss = function(y_true, y_hat, tau){
  return(mean((tau-(y_true < y_hat)) * (y_true-y_hat)))
}

# cria um vetor vazio do tamanho dos dados de treino
VaR<-rep(NA,nrow(train)) 

# inicia o primeiro VaR para o modelo adaptativo
VaR[1]<- -qnorm(tau)*sd(train[, ativo]) 

# Modelo Adaptativo
# VaR_t = VaR_t-1 + beta * hit
# hit = I(Y_t-1 < -VaR_t-1) - tau
adpatative <- function(beta){
    
  for(i in 2:nrow(train)){
    VaR[i] <- VaR[i-1] + beta*((train[i-1, ativo] < -VaR[i-1])-tau)
  }
        
  # Função objetivo, ou custo da regressão quantílica
  return(loss(train[, ativo], VaR, tau))
}

# Otimização da função custo no parâmetro beta
res <- optim(par=c(0),adpatative,method="Brent",lower=c(-100),upper=c(100))
beta <- res$par # fixa o beta ótimo

# Faz a previsão utilizando o modelo adapptativo: VaR_t = VaR_t-1 + beta * hit
adpatativeForecast<-function(beta, data){
  for(i in 2:nrow(data)){
    VaR[i] <- VaR[i-1] + ((data[i-1, ativo]< -VaR[i-1])-tau)*beta
  }
  return(VaR)
}

# VaR de treino
VaR_train = adpatativeForecast(beta, train) # fixa os VaRs estimados
train["VaR"] = VaR_train

p = ggplot(train) + 
  geom_line(aes(x=Date, y=Adjusted.Close, colour = "Return"))+
  geom_line(aes(x=Date, y=-VaR, colour = "VaR"))+ 
  xlab("")
p

# VaR de validação
# inicia o primeiro VaR para o modelo adaptativo
VaR<-rep(NA,nrow(valid)) 
VaR[1]<- -qnorm(tau)*sd(valid[, ativo]) 

VaR_valid <- adpatativeForecast(beta, valid) # fixa os VaRs estimados
valid["VaR"] = VaR_valid

p = ggplot(valid) + 
  geom_line(aes(x=Date, y=Adjusted.Close, colour = "Return"))+
  geom_line(aes(x=Date, y=-VaR, colour = "VaR"))+ 
  xlab("")
p

print(paste('Número Hits de treino', mean(train[, ativo] < -VaR_train))) # avalia o VaR de treino como n# de hits (quanto mais próximo de tau melhor)
print(paste('Número Hits de Validação', mean(valid[, ativo] < -VaR_valid))) # avalia o VaR de treino como n# de hits (quanto mais próximo de tau melhor)

print(paste('Custo de Treino', loss(train[, ativo], -VaR_train, tau) )) # avalia o VaR de treino como n# de hits (quanto mais próximo de tau melhor)
print(paste('Custo de Validação', loss(valid[, ativo], -VaR_valid, tau) )) # avalia o VaR de treino como n# de hits (quanto mais próximo de tau melhor)


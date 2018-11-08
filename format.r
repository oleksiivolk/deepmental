library(foreign) 
write.table(read.spss("data.sav"), file="data.csv", quote = TRUE, sep = ",")

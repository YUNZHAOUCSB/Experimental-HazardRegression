gname="model"
gf = file(gname, open="r")
gLines=readLines(gf, n=10)
ename = "toy_100"
ef = file(ename, open="r")
eLines = readLines(ef, n=10)


for(i in 1:length(gLines)) {
  golden = unlist(strsplit(gLines[i], split='\t'))
  est = unlist(strsplit(eLines[i], split='\t'))
  a = golden[2:length(golden)]
  b = est[2:length(est)]
  x1=vector(mode="numeric", length=0)
  y1=vector(mode="numeric", length=0)
  x2=vector(mode="numeric", length=0)
  y2=vector(mode="numeric", length=0)
  x1=c(x1,0)
  y1=c(y1,0)
  x2=c(x2,0)
  y2=c(y2,0)
  for(j in 1:length(a)) {
    sub=a[j]
    sub=unlist(strsplit(sub,split=':'))
    x1 = c(x1, sub[1])
    y1 = c(y1, sub[2])
  }
  for(j in 1:length(b)) {
    sub=b[j]
    sub=unlist(strsplit(sub,split=':'))
    x2 = c(x2, sub[1])
    y2 = c(y2, sub[2])
  }
  x1=as.numeric(x1)
  y1=as.numeric(y1)
  x2=as.numeric(x2)
  y2=as.numeric(y2)
  pdf(sprintf("feat%d.pdf", i-1))
  #if(i==2) plot(x1,y1,type='s',xlab="time",ylab="hazard rate",col="red",xlim=c(0,10),ylim=c(0,0.27),lwd=c(1.5,1.5),cex.lab=1.5)
  #else if(i==4) plot(x1,y1,type='s',xlab="time",ylab="hazard rate",col="red",xlim=c(0,10),ylim=c(0,0.317),lwd=c(1.5,1.5),cex.lab=1.5)
  #plot(x1,y1,type='s',xlab="time",ylab="hazard rate",col="red",xlim=c(0,10),ylim=c(0,6e-1),lwd=c(1.5,1.5),cex.lab=1.7,cex.axis=1.7)
  plot(x1,y1,type='s',xlab="time",ylab="hazard rate",col="red",xlim=c(0,10),lwd=c(1.5,1.5),cex.lab=1.7,cex.axis=1.7)
  lines(x2,y2,type='s',col="blue", lwd=c(1.5,1.5))
  legend("topleft",c("ground truth","estimate"),lty=c(1,1),lwd=c(3,3),col=c("red","blue"),cex=1.6)
}
close(gf)
close(ef)

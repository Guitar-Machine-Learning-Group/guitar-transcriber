#Find the true/false positive/negative values
#Yt is the training data (ours)
#Y is the target data (the correct)
def PosNeg(Y,Yt):
	m = 51
	n = len(Y)
	TN = 0 #true negative
	TP = 0 #true positive
	FP = 0 #False positive
	FN = 0 #False negative
	for i in range(0,n):
		y = Y[i] 
		yt = Yt[i] 
		for j in range(0,m):
			a = y[j]
			b = yt[j]
			if a==b:
				if a==1:
					TP+=1
				else:
					TN+=1
			else:
				if b==1:
					FP+=1
				else:
					FN+=1
	A= [TP, TN, FP, FN]
	return A

#return the fmeasure value of the two Ys
def fmeasure(Y,Yt):
	A = PosNeg(Y,Yt)
	TP = A[0]
	FP = A[2]
	FN = A[3]
	return 2.0*(float(TP)/float(2*TP+FN+FP))

# Y = [[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0]]
# Yt =[[1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1]]

# Yt =[[1,1,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1]]
# print(fmeasure(Y,Y))
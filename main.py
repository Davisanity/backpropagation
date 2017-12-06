from neural import Neural

def main():
	input = [(1,2),(-1,1)]
	output = [(0.3,0.6),(0.6,0.8)]
	n = Neural((1,2),(3,5),2,1,3,2)
	print ("w_h:",n.w_h)
	print ("w_o:",n.w_o)
	for i in range(10):
		# print("*****"+str(i)+"*****")
		n.forword()
		n.errorFunc()
		n.sigma()
		n.updateWeight()
		print ("ERR:",n.errorFunc())
		if(n.errorFunc()<8.455):
			print("iteration "+str(i)+" times")
			break

		print ("updated w_h",n.w_h)
		print ("updated w_o",n.w_o)
	# print ("updated w_ih",n.w_ih)
	# print ("updated w_h",n.w_h)
	# print ("updated w_ho",n.w_ho)
	# print ("updated w_o",n.w_o)
	print ("output",n.output)



if __name__ == "__main__":
    main()

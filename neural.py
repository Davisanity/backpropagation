import random
import math

#implement 2D empty list
# list=[]
# for i in range(10):
     # new=[]
     # for i in range(10):
            # new.append(None)
     # list.append(new)

class Neural:

	def __init__(self,input,output,i,layerNum_h,h,o):
		self.input = input
		self.hidden= [[None for _ in range(h)] for _ in range(layerNum_h)]
		self.Output = output
		self.output=[None]*o
		self.num_i=i
		self.lnum_h=layerNum_h
		self.num_h=h
		self.num_o=o

		w_i=[]
		w_h=[[None for _ in range(h)] for _ in range(layerNum_h)]
		w_o=[]
		w_ih=[]
		w_hh=[[None for _ in range(h)] for _ in range(layerNum_h)]
		w_ho=[]
		self.learn = 3#learning rate
		self.sigma_h=[[None for _ in range(h)] for _ in range(layerNum_h)]
		self.sigma_o=[None]*o

		self.randWeight(i,layerNum_h,h,o)
		# self.forword()
		# self.errorFunc()
		# self.sigma()
		# self.updateWeight()
	def rand(self):
		a = float('%.2f'%random.uniform(-1,3))
		while a == 0 :
			a = float('%.2f'%random.uniform(-1,3))
		return a

	def randWeight(self,input,layerNum_h,hidden,output):
		# r=('%.2f'%random.uniform(0,1))
		# self.w_i = [self.rand() for h in range(input)]
		self.w_h=[[ float('%.2f'%random.uniform(-2,-1)) for h in range(hidden)] for i in range(layerNum_h)]
		self.w_o = [ float('%.2f'%random.uniform(-2,-1)) for h in range(output)]
		self.w_ih = [[ self.rand() for h in range(hidden)] for i in range(input)]
		# hidden layer's weight.But last layer is connected to output layer so layerNum_h-1
		if layerNum_h>1:
			self.w_hh=[[ self.rand() for h in range(hidden)] for i in range(layerNum_h-1)]
		self.w_ho = [[ self.rand() for o in range(output)] for h in range(hidden)]

		# print(type(self.w_ih[0][0])) #str

	def forword(self):
		for h in range(self.num_h):
			t=0
			for i in range(self.num_i):
				t += float(self.input[i])*float(self.w_ih[i][h])
			sum =  t + self.w_h[0][h]
			self.hidden[0][h] = float('%.5f'%float(self.activeFunc(sum)) )
		
		if self.lnum_h>1:
			for l in range(self.lnum_h):
				for h in range(self.num_h):
					t=0
					for p in range(self.num_h):   #p means prev hidden layer
						t+= float(self.hidden[p][h])*float(self.w_hh[l][h])
					sum = t+self.w_h[l][h]
				self.hidden[l][h]= float('%.5f'%float(self.activeFunc(sum)) )
		
		for o in range(self.num_o):
			t=0
			for h in range(self.num_h):
				t += float(self.hidden[self.lnum_h-1][h])*float(self.w_ho[h][o])
			sum = t + self.w_o[o]
			self.output[o] = float( '%.5f'%float(self.activeFunc(sum)) )
		# print("hidden")
		# print self.hidden
		# print self.output


	def activeFunc(self,sum):
		func = 1.0/(1+math.exp(sum*(-1)))
		return func

	def errorFunc(self):
		sum=0
		for i in range(self.num_o):
			sum += (self.Output[i]-float(self.output[i]))**2
		# print ("sum/2",sum/2)
		return sum/2

	def sigma(self):
		#caculate output's sigma
		for o in range(self.num_o):
			s=(self.Output[o]-self.output[o])*self.output[o]*(1-self.output[o])
			self.sigma_o[o]=float('%.5f'%s)
		#caculate hidden's sigma
		for l in range(self.lnum_h-1,-1,-1):
			for h in range(self.num_h):
				t=0
				for o in range(self.num_o):
					t+=self.sigma_o[o]*self.w_ho[h][o]
				self.sigma_h[l][h]=float('%.5f'% ( t*self.hidden[l][h]*(1-self.hidden[l][h]) ) )
				if self.lnum_h>1 and l != self.lnum_h-1 :
					for k in range(self.num_h):
						self.sigma_h[l][h] += self.sigma_h[l+1][k]*self.w_hh[l][k]
					self.sigma_h[l][h] = float('%.5f'% self.sigma_h[l][h])

		# print ("sigma_h",self.sigma_h)
		# print ("sigma_o",self.sigma_o)
		# return self.sigma_h,self.sihma_o

#2017 1204 tclin  HERE  NEED TO THINK MORE
	def updateWeight(self):
		delta_o = []
		delta_h = [[None for _ in range(self.num_h)] for _ in range(self.lnum_h)]
		
		#caculate w_o and w_h
		for o in range(self.num_o):
			delta_o.append(float('%.4f'%(self.learn*self.sigma_o[o])))
			self.w_o[o]+=delta_o[o]
			self.w_o[o] = float('%.3f'%self.w_o[o] )
		for l in range(self.lnum_h):
			for h in range(self.num_h):
				delta_h[l][h] = float('%.4f'%(self.learn*self.sigma_h[l][h]))
				self.w_h[l][h]+=delta_h[l][h]
				self.w_h[l][h] = float('%.3f'%self.w_h[l][h] )
		#update w_ho 
		for h in range(self.num_h):
			for o in range(self.num_o):
				delta_ho = self.learn*(self.Output[o]-self.output[o])*self.output[o]*(1-self.output[o])*self.hidden[self.lnum_h-1][h]
				self.w_ho[h][o]+=float('%.4f'%delta_ho)
				self.w_ho[h][o]=float('%.3f'%self.w_ho[h][o])
		#update w_hh
		for l in range(self.lnum_h-2,-1,-1):
			for h in range(self.num_h):
				delta_hh = self.learn*self.hidden[l+1][h]*(1-self.hidden[l+1][h])*self.sigma_h[l+1][h]*self.w_hh[l][h]*self.hidden[l][h]
				self.w_hh[l][h]+=float('%.4f'%delta_hh)
				self.w_hh[l][h]=float('%.3f'%self.w_hh[l][h])
		#update w_ih
		for i in range(self.num_i):
			for h in range(self.num_h):
				delta_ih = self.learn*self.hidden[0][h]*(1-self.hidden[0][h])*self.sigma_h[0][h]*self.input[i]	
				self.w_ih[i][h]+=float('%.4f'%delta_ih)
				self.w_ih[i][h]=float('%.3f'%self.w_ih[i][h])
		
		# print ("updated w_h",self.w_h)
		# print ("updated w_o",self.w_o)
		# print self.w_ho
		# print self.w_ih

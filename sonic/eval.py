import matplotlib.pyplot as plt
import numpy as np

def main():
	global epsilons
	global epsilons4
	global epsilons1
	epsilons = ['Act 1 \u03B5= 0.3','Act 1 \u03B5= 0.5','Act 1 \u03B5= 0.7','Act 2 \u03B5= 0.3',' Act 2 \u03B5= 0.5','Act 2 \u03B5= 0.7','Act 3 \u03B5= 0.3','Act 3 \u03B5= 0.5','Act 3 \u03B5= 0.7']
	epsilons4= ['Act 1 \u03B5= 0.3','Act 1 \u03B5= 0.5','Act 1 \u03B5= 0.7','Act 2 \u03B5= 0.3',' Act 2 \u03B5= 0.5','Act 2 \u03B5= 0.7']
	epsilons1 = ['Act 2 \u03B5= 0.3',' Act 2 \u03B5= 0.5','Act 2 \u03B5= 0.7','Act 3 \u03B5= 0.3','Act 3 \u03B5= 0.5','Act 3 \u03B5= 0.7']

	Lvl1_complete=[8032,8032,8032,10538,10538,10538]
	Lvl1=[4289,4273,4264, 7508,7477,7573]
	com1=[]
	nonCom1=[]
	for x in range(len(Lvl1)):
		com1.append(Lvl1[x]/Lvl1_complete[x])
		nonCom1.append(1-Lvl1[x]/Lvl1_complete[x])
	plotIt1(com1,nonCom1,"Level 1, Greenhill Zone")

	Lvl2_complete=[6736,6736,6736,4432,4432,4432,7364,7364,7364]
	Lvl2=[3317,3317,3317,1525,1525,1525,2202,2202,2202]
	com2=[]
	nonCom2=[]
	for x in range(len(Lvl2)):
		com2.append(Lvl2[x]/Lvl2_complete[x])
		nonCom2.append(1-Lvl2[x]/Lvl2_complete[x])
	plotIt(com2,nonCom2,"Level 2, Labyrinth Zone")

	Lvl3_complete=[6240,6240,6240,6240,6240,6240,6240,5920,5920,5920]
	com3=[]
	nonCom3=[]
	Lvl3=[3957,3957,3957,1653,1653,1653,2037,2037,2037]
	for x in range(len(Lvl3)):
		com3.append(Lvl3[x]/Lvl3_complete[x])
		nonCom3.append(1-Lvl3[x]/Lvl3_complete[x])
	plotIt(com3,nonCom3,"Level 3, Marble Zone")
	
	Lvl4_complete=[8288,8288,8288,8288,8008,8008,8008]
	com4=[]
	nonCom4=[]
	Lvl4=[2806,2290,2308,1141,1941,1141]
	for x in range(len(Lvl4)):
		com4.append(Lvl4[x]/Lvl4_complete[x])
		nonCom4.append(1-Lvl4[x]/Lvl4_complete[x])
	plotIt4(com4,nonCom4,"Level 4, ScrapBrain Zone")
	
	Lvl5_complete=[9056,9056,9056,10592,10592,10592,11139,11139,11139]
	com5=[]
	nonCom5=[]
	Lvl5=[3291,4639,4179,2998,4752,3925,3574,3574,3574]
	for x in range(len(Lvl5)):
		com5.append(Lvl5[x]/Lvl5_complete[x])
		nonCom5.append(1-Lvl5[x]/Lvl5_complete[x])
	plotIt(com5,nonCom5,"Level 5, Spring Yard Zone")
	
	Lvl6_complete=[8800,8800,8800,7904,7904,7904,7904,7904,7904]
	com6=[]
	nonCom6=[]
	Lvl6=[4453,4365,4514,3528,6649,3486,3072,3232,3279]
	for x in range(len(Lvl6)):
		com6.append(Lvl6[x]/Lvl6_complete[x])
		nonCom6.append(1-Lvl6[x]/Lvl6_complete[x])
	plotIt(com6,nonCom6,"Level 6, Star Light Zone")

def plotIt(com,nonCom,lvl):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	y = np.arange(len(epsilons))
	ax.barh(y, com, align='center', height=.25, color='#00ff00',label='completed')
	ax.barh(y, nonCom, align='center', height=.25, left=com, color='red',label='not completed')
	ax.set_yticks(y)
	ax.set_yticklabels(epsilons)
	ax.set_xlabel('Percentege of Level finished')
	ax.set_title(lvl)
	ax.grid(False)
	ax.legend()
	plt.tight_layout()
	#plt.savefig(lvl+'stackedbar.png')
	plt.show()
def plotIt4(com,nonCom,lvl):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	y = np.arange(len(epsilons4))
	ax.barh(y, com, align='center', height=.25, color='#00ff00',label='completed')
	ax.barh(y, nonCom, align='center', height=.25, left=com, color='red',label='not completed')
	ax.set_yticks(y)
	ax.set_yticklabels(epsilons4)
	ax.set_xlabel('Percentege of Level finished')
	ax.set_title(lvl)
	ax.grid(False)
	ax.legend()
	plt.tight_layout()
	#plt.savefig(lvl+'stackedbar.png')
	plt.show()
def plotIt1(com,nonCom,lvl):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	y = np.arange(len(epsilons1))
	ax.barh(y, com, align='center', height=.25, color='#00ff00',label='completed')
	ax.barh(y, nonCom, align='center', height=.25, left=com, color='red',label='not completed')
	ax.set_yticks(y)
	ax.set_yticklabels(epsilons1)
	ax.set_xlabel('Percentege of Level finished')
	ax.set_title(lvl)
	ax.grid(False)
	ax.legend()
	plt.tight_layout()
	#plt.savefig(lvl+'stackedbar.png')
	plt.show()

if __name__ == "__main__":
    main()
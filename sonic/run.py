import sonic_multi_input

#run multiple trainings
def main():
	#evaluation Training sets -> currently 72 trainings
	epsilonList=[0.3, 0.5, 0.7]
	experimentList=[2000]
	timeStepsList=[5000,10000,15000]
	#mb_sizeList=[32,128]
	mb_sizeList=[32]
	frameStacksList=[4]
	#epsilonList=[0.1, 0.3, 0.5, 0.7]
	#experimentList=[2000]
	#timeStepsList=[5000,10000,15000]
	#mb_sizeList=[32,64,128]
	#frameStacksList=[4,6]

	for e in epsilonList:
		for ex in experimentList:
			for s in timeStepsList:
				if s==5000:
					exp=2000
				if s==10000:
					exp=1000
				if s== 15000:
					exp=660
				for mb in mb_sizeList:
					for f in frameStacksList:
						sonic_multi_input.main(e,exp,s,mb,f)

if __name__ == "__main__":
    main()

import sonic_multi_input

#run multiple trainings
def main():
	#evaluation Training sets -> currently 72 trainings
	epsilonList=[0.1, 0.3, 0.5, 0.7]
	experimentList=[2000]
	timeStepsList=[5000,10000,15000]
	mb_sizeList=[32,64,128]
	frameStacksList=[4,6]

	for e in epsilonList:
		for ex in experimentList:
			for s in timeStepsList:
				for mb in mb_sizeList:
					for f in frameStacksList:
						sonic_multi_input.main(e,ex,s,mb,f)

if __name__ == "__main__":
    main()

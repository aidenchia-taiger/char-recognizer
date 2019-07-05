import keras
import argparse
from model import ModelFactory

def main():
	mf = ModelFactory(batchSize=32, numClasses=47, imgSize=(28,28,1), dropoutRatio=0.6, modelName=None,
				 filterVals=[16,32,64], kernelVals=[4,4,4], poolVals=[2,2,2], strideVals=[1,1,1],
				 learningRate=0.001, validation_split=0.1, numEpochs=50)


if __name__ == "__main__":
	main()

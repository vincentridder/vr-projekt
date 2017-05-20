#!/usr/local/bin/python2
from pyAudioAnalysis import audioTrainTest as aT
import os
from sys import argv
script, dirname = argv

subdirectories = os.listdir(dirname)

subdirectories = [dirname + "/" + subDirName for subDirName in subdirectories]

aT.featureAndTrain(subdirectories, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmModel", False)



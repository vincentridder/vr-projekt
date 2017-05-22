from record_samples import record
import numpy as np
import wave
import time 
from run import check_sample
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO as aIO

import os
from sys import argv

modelType = "svm"
modelName = "svmModel"

[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadSVModel(modelName)



skript, dirname = argv

subdirs = os.listdir(dirname)
classNames2 = subdirs
subdirs = [dirname + "/" + subdir for subdir in subdirs]

num_right = 0
time_t = 0
num_files = 0
for subdir in subdirs:
    print(subdir)
    for wav_path in os.listdir(subdir):
        num_files+=1
        print("test "+wav_path+ ":")
        # load audio data
        Fs, x = aIO.readAudioFile(subdir+"/"+wav_path)
        t_s = time.time()
        print (Fs)
        winner = check_sample(x, Fs, mtWin, mtStep, stWin, stStep, Classifier,  modelType, computeBEAT, MEAN, STD)

        t_e = time.time()
        time_h= t_e - t_s
        print("winner: " + str(classNames[winner]))
        print("time : "+ str(time_h))
        time_t +=time_h
        if (subdir == dirname+ "/" +classNames[winner]):
            num_right += 1.0

print( "% richtig : " + str(num_right/ num_files))
print( "avarage check time = " + str(time_t/num_files))
print(end)




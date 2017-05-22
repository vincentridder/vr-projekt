from record_samples import record
import numpy as np
import pyaudio
import wave
import time
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioFeatureExtraction as aF

chunk=1024
isSignificant = 0.5 #try different values.


def check_sample(signal, Fs, mtWin, mtStep, stWin, stStep,Classifier, modelType, computeBEAT, MEAN, STD):

    # feature extraction:
    [MidTermFeatures, s] = aF.mtFeatureExtraction(signal, Fs, mtWin * Fs, mtStep * Fs, round(Fs * stWin), round(Fs * stStep))
    MidTermFeatures = MidTermFeatures.mean(axis=1)        # long term averaging of mid-term statistics
    
    if computeBEAT:
        [beat, beatConf] = aF.beatExtraction(s, stStep)
        MidTermFeatures = numpy.append(MidTermFeatures, beat)
        MidTermFeatures = numpy.append(MidTermFeatures, beatConf)

    # feature extraction:
    [MidTermFeatures, s] = aF.mtFeatureExtraction(signal, Fs, mtWin * Fs, mtStep * Fs, round(Fs * stWin), round(Fs * stStep))
    MidTermFeatures = MidTermFeatures.mean(axis=1)        # long term averaging of mid-term statistics
    
    if computeBEAT:
        [beat, beatConf] = aF.beatExtraction(s, stStep)
        MidTermFeatures = numpy.append(MidTermFeatures, beat)
        MidTermFeatures = numpy.append(MidTermFeatures, beatConf)
    curFV = (MidTermFeatures - MEAN) / STD                # normalization

    [Result, P] = aT.classifierWrapper(Classifier, modelType, curFV)    # classification        
                                                    
    
    winner = np.argmax(P) #pick the result with the highest probability value.
    if P[winner] > isSignificant :
        return winner
    

    
def play_sound( path):
#open a wav format music  
    f = wave.open(path,"rb")  
#instantiate PyAudio  
    p = pyaudio.PyAudio()  
#open stream  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                        channels = f.getnchannels(),  
                                        rate = f.getframerate(),  
                                                        output = True)  
#read data  
    data = f.readframes(chunk)  

#play stream  
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  

#stop stream  
    stream.stop_stream()  
    stream.close()  

#close PyAudio  
    p.terminate() 

if __name__ == '__main__':
    modelType = "svm"
    modelName ="svmModel"
    #   todo : hinzufuegen anderer clasifiere
    #   load classifier: 
    [Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadSVModel(modelName)
    
    Fs = 44100
    while True :
        
        sample_width, signal = record(10,24)               
        t_start=time.time()
        
        winner = check_sample(signal, Fs, mtWin, mtStep, stWin, stStep,Classifier, modelType, computeBEAT, MEAN, STD)
        
        t_end = time.time()
        
        print("time for classifikation= " + str(t_end
        -t_start))
       
        print(classNames[winner])
        





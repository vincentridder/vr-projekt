from record_samples import record_to_file
import numpy as np
import pyaudio
import wave
import time

from pyAudioAnalysis import audioTrainTest as aT
chunk=1024
isSignificant = 0.5 #try different values.

def check_sample(path):
    Result, P, classNames = aT.fileClassification(path, "svmModel", "svm")
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
    
    while True :
        
        record_to_file('0.wav',10,24)               
        
        t_start=time.time()
        winner=check_sample("0.wav")
        t_end = time.time()
        
        print("time for classifikation= " + str(t_end
        -t_start))
       
        print(winner)
        


        if winner == 0:

            play_sound("1.wav")

        elif winner == 1:
            play_sound("2.wav")



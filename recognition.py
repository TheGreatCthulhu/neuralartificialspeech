import os 
import pickle 
import numpy as np 
from scipy.io.wavfile import read 
import python_speech_features as mfcc 
from sklearn import preprocessing 
import warnings 
import timeit 
warnings.filterwarnings("ignore") 


nfft=2048 
appendEnergy = False 
def get_MFCC(sr,audio): 
features = mfcc.mfcc(audio,sr, 0.025, 0.01, 13, 26, nfft, 0, 1000, appendEnergy) 
features = preprocessing.scale(features) return features 

#path to test data 
sourcepath = "C:/NYC/Test/Syntez/" 
#path to saved models 
modelpath = "C:/NYC/Models/" 

gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')] 
models = [pickle.load(open(fname,'rb')) for fname in gmm_files] 
genders = [fname.split('/')[-1].split('.gmm')[0] for fname in gmm_files] 
files = [os.path.join(sourcepath,f) for f in os.listdir(sourcepath) 
if f.endswith('.wav') or f.endswith('.wave')] 
gender=[] 
bad_names=[] 
for f in files: 
start = timeit.default_timer() 
print(f.split("/")[-1]) 
sr, audio = read(f) 
features = get_MFCC(sr,audio) #old 
scores = None 
log_likelihood = np.zeros(len(models)) 
for i in range(len(models)): 
gmm = models[i] #checking with each model one by one 
scores = np.array(gmm.score(features)) 
log_likelihood[i] = scores.sum() 
winner = np.argmax(log_likelihood) 
gender.append(winner) 
if winner==1: 
bad_names.append(f.split('/')[-1]) 
print("\tdetected as - ", genders[winner],"\n\tscores:original ",log_likelihood[0],",sintez ", log_likelihood[1],"\n") 
stop = timeit.default_timer() 
print(stop-start)

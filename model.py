import os 
import pickle 
import numpy as np 
from scipy.io.wavfile import read 
from sklearn.mixture import GaussianMixture as GMM 
from python_speech_features import mfcc 
from sklearn import preprocessing 
import warnings 
warnings.filterwarnings("ignore") 

nfft=2048 
appendEnergy = False 
def get_MFCC(sr,audio): 
features = mfcc(audio,sr, 0.025, 0.01, 13, 26, nfft, 0, 1000, appendEnergy) #best 1000  
features = preprocessing.scale(features) 
return features 


#path to training data 
source = '/NYC/Sintez/' 
#path to save trained model 
dest = "/NYC/Models/" 
files = [os.path.join(source,f) for f in os.listdir(source) if 
f.endswith('.wav') or f.endswith('.wave')] 
features = np.asarray(()); 

for f in files: 
sr,audio = read(f) 
vector = get_MFCC(sr,audio) #old 
if features.size == 0: 
features = vector 
else: 
features = np.vstack((features, vector)) 

gmm = GMM(n_components = 8, covariance_type='diag', n_init = 3) 
gmm.fit(features) 
picklefile = f.split("/")[-2].split(".wav")[0]+".gmm" 

# model saved as male.gmm 
with open(dest+picklefile,'wb') as fout: 
pickle.dump(gmm,fout) 
print('modeling completed for speach:',picklefile)

"""
Copied this file brutally from a team member for reading in the sounds
Replacing hand-rolled learning with keras deep learning

"""

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
import scipy.io.wavfile as siw

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical



pathes = ['en01.wav', 'de01.wav', 'en02.wav', 'de02.wav']
#pathes = ['en01.wav']
#pathes = ['de01.wav']
          
samples = []
samplerates = []

for path in pathes:
    audio = siw.read(path)
    samples.append(audio[1])
    samplerates.append(audio[0])
    samples[-1] = np.divide(samples[-1],max(abs(samples[-1])))


""" 
commenting out stuff for his learning

#   RESERVOIR
r_size = 100
w = (np.random.rand(r_size, r_size)*2-1)**5
spec = np.max(np.abs(np.linalg.eigvals(w)))
w /= spec

#   UPDATE ROUTINE
updates = 10000
out_size = 3
res_hist = []
out_idx = np.random.randint(0,len(w),out_size)
trajectory_data = []
"""
model = Sequential()
model.add(Dense(64, input_dim=100000))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, input_dim=100000))
model.add(Activation('tanh'))
model.add(Dense(3))
model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['accuracy'])

samples = list(map(lambda s: s[0:100000], samples))
samples = np.column_stack(samples).T

labels = to_categorical([1, 2, 1, 2])

#pairs = [lambda (l, s):
#            l 
#        for (l, s) in zip(labels, samples)]

model.fit(np.asarray(samples), labels, nb_epoch=10)

"""
    print(len(sample))
    act = np.zeros(r_size)
    out = np.zeros(out_size)
    out_hist = []
    for i in range(updates):
        act[:5] = sample[updates]    
        res_hist.append(act)
        out_hist.append(out)
        act = np.tanh(w @ act)
        out = [act[i] for i in out_idx]
        
    trajectory_data.append(np.array(out_hist))
    """
"""
#   TRAJECTORY PLOT
fig1 = plt.figure()

ax = fig1.gca(projection='3d')

#   READOUT
classes = 2
targets = [1,0]
readouts = []
for i in range(classes):
    idxs=np.random.randint(0,len(w),out_size)
    data=np.array([[a[j] for j in idxs] for a in res_hist])
    #readouts += (idxs, Perceptron().fit(data,targets))
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    ax.plot(x,y,z)
    ax.scatter(x,y,z)
plt.show()

#plt.imshow(res_hist,interpolation='nearest')
#plt.imshow(out_hist,interpolation='nearest')
"""

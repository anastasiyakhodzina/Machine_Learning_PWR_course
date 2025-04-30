# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""


#### Libraries
# Standard library
import pickle
import gzip
import math

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def fi(x):
    return 1/(1+np.exp(-x))

def fi_derative(x):
    return fi(x)*(1-fi(x))


train_data,val_data,test_data=load_data_wrapper()
#len(train_data)=50000
#x,y=train_data[0]

X1_train=[np.vstack((np.array([[1]]), x))  for x,y in train_data] #X_train with added ones
X1_val=[np.vstack((np.array([[1]]), x))  for x,y in val_data] #X_validation with added ones

Y_train=[y  for x,y in train_data] #Y
Y_val=[y  for x,y in val_data]

n=785
W1=np.random.normal(loc=0,scale=math.sqrt(2/n),size=(64,785)) #Weight №1
W2=np.random.normal(loc=0,scale=math.sqrt(2/n),size=(32,65)) #Weight №2
W3=np.random.normal(loc=0,scale=math.sqrt(2/n),size=(16,33)) #Weight №3
W4=np.random.normal(loc=0,scale=math.sqrt(2/n),size=(10,17)) #Weight №4

weights=[W1,W2,W3,W4]
weights_=[W1,W2,W3,W4] #weights to change
c=0.415 #learning rate

for j in range(0,30): # 30 epochs
    for i in range (0,50000-2,3):#50000 training images
        #forwardpropagation
        Xk1=[X1_train[i]]# batch learning for 3 inputs in one iterations
        Xk2=[X1_train[i+1]]
        Xk3=[X1_train[i+2]]
        WkXk1=[]
        WkXk2=[]
        WkXk3=[]
        ak1=[]
        ak2=[]
        ak3=[]
        for k in range(0,len(weights)):
            WkXk1.append(np.dot(weights[k],Xk1[k]))
            ak1.append(fi(WkXk1[k]))
            if k != len(weights)-1: Xk1.append(np.vstack((np.array([[1]]),ak1[k])))

            WkXk2.append(np.dot(weights[k],Xk2[k]))
            ak2.append(fi(WkXk2[k]))
            if k != len(weights)-1: Xk2.append(np.vstack((np.array([[1]]),ak2[k])))

            WkXk3.append(np.dot(weights[k],Xk3[k]))
            ak3.append(fi(WkXk3[k]))
            if k != len(weights)-1: Xk3.append(np.vstack((np.array([[1]]),ak3[k])))

        #backpropagation
        for k in range(len(weights)-1,-1,-1):
            
          if k == len(weights)-1:
              dL_dak1=ak1[k]-Y_train[i]
              dL_dak2=ak2[k]-Y_train[i]
              dL_dak3=ak3[k]-Y_train[i]
          else:
              dL_dak1=dl_dXk1[1:]
              dL_dak2=dl_dXk2[1:]
              dL_dak3=dl_dXk3[1:]
        
          sygnal_deltak1=dL_dak1*fi_derative(WkXk1[k])
          sygnal_deltak2=dL_dak2*fi_derative(WkXk2[k])
          sygnal_deltak3=dL_dak3*fi_derative(WkXk3[k])

          dl_dWk1=np.dot(sygnal_deltak1,np.transpose(Xk1[k]))
          dl_dWk2=np.dot(sygnal_deltak2,np.transpose(Xk2[k]))
          dl_dWk3=np.dot(sygnal_deltak3,np.transpose(Xk3[k]))
        
          #changing weights
          weights_[k]=weights[k]-c*(dl_dWk1+dl_dWk2+dl_dWk3)/3#in batch learning dl_dwk is sum of elements , so divide by batch size

          dl_dXk1=np.dot(np.transpose(weights[k]),sygnal_deltak1)
          dl_dXk2=np.dot(np.transpose(weights[k]),sygnal_deltak2)
          dl_dXk3=np.dot(np.transpose(weights[k]),sygnal_deltak3)

        weights=weights_.copy()


right_guesses=0
for i in range (0,10000):#10000-val_data
    ak=0
    for k in range(0,len(weights)):
        if k ==0:
            WkXk=np.dot(weights[k],X1_val[i])
        else:
            WkXk=np.dot(weights[k],Xk)
        ak=fi(WkXk)
        if k != len(weights)-1: Xk=np.vstack((np.array([[1]]),ak))  
    if np.argmax(ak)==Y_val[i]: right_guesses+=1

#probability(right guess)
print('probability(right guess) = ',right_guesses/10000)#after one try i got prob=0.9101
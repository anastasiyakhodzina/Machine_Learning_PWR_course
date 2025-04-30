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
    for i in range (0,50000):#50000
        #forwardpropagation
        Xk=[X1_train[i]]
        WkXk=[]
        ak=[]
        for k in range(0,len(weights)):
            WkXk.append(np.dot(weights[k],Xk[k]))
            ak.append(fi(WkXk[k]))
            if k != len(weights)-1: Xk.append(np.vstack((np.array([[1]]),ak[k])))

        #backpropagation
        for k in range(len(weights)-1,-1,-1):
            
          if k == len(weights)-1:
              dL_dak=ak[k]-Y_train[i]
          else:
              dL_dak=dl_dXk[1:]
        
          sygnal_deltak=dL_dak*fi_derative(WkXk[k])

          dl_dWk=np.dot(sygnal_deltak,np.transpose(Xk[k]))
        
          #changing weights
          weights_[k]=weights[k]-c*dl_dWk#in batch learning dl_dwk is sum of elements , so divide by batch size

          dl_dXk=np.dot(np.transpose(weights[k]),sygnal_deltak)

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
print('probability(right guess) = ',right_guesses/10000)#after one try i got prob=0.9657

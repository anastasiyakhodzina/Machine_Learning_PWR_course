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
#print(x,'\n\n\n\n\n',y)
X1_train=np.array([np.vstack((np.array([[1]]), x))  for x,y in train_data]) #X with added ones
X1_val=np.array([np.vstack((np.array([[1]]), x))  for x,y in val_data])

Y_train=np.array([y  for x,y in train_data]) #Y
Y_val=np.array([y  for x,y in val_data])

n=785
W1=np.random.normal(loc=0,scale=math.sqrt(2/n),size=(64,785)) #Weight №1
W2=np.random.normal(loc=0,scale=math.sqrt(2/n),size=(32,65)) #Weight №2
W3=np.random.normal(loc=0,scale=math.sqrt(2/n),size=(10,33)) #Weight №3

c=0.415#dla c = 0.1 prawdopodobienstwo(trafienia)=0.9212, a dla c = 0.415 - 0.9648 
# dodac epoki for j in range(0,30):
for j in range(0,30): # 30 epok
    for i in range (0,50000):#50000
        #pierwsza warstwa
        W1X1=np.dot(W1,X1_train[i])
        a1=fi(W1X1)
        X2=np.vstack(( np.array([[1]]),a1))
        
        #druga warstwa
        W2X2=np.dot(W2,X2)
        a2=fi(W2X2)
        X3=np.vstack(( np.array([[1]]),a2))

        #trzecia warstwa
        W3X3=np.dot(W3,X3)
        a3=fi(W3X3)

        #propagacja wstecz
        dL_da3=a3-Y_train[i]
        
        sygnal_delta3=dL_da3*fi_derative(W3X3)#np.array([x*y for x in dL_da3 for y in fi_derative(W3X3) ][::(len(dL_da3)+1)]) # Hadamart product albo dL_da3*fi_derative(W3X3)

        dl_dW3=np.dot(sygnal_delta3,np.transpose(X3))
        
        #zmiana W3
        W3_=W3-c*dl_dW3

        dl_dX3=np.dot(np.transpose(W3),sygnal_delta3)

        dl_da2=dl_dX3[1:]#usuniecie peirwszej spolrzednej opowiadajacej wejsciu=1
        
        sygnal_delta2=dl_da2*fi_derative(W2X2)

        dl_W2=np.dot(sygnal_delta2,np.transpose(X2))

        #zmiana W2
        W2_=W2-c*dl_W2

        dl_dX2=np.dot(np.transpose(W2),sygnal_delta2)

        dl_da1=dl_dX2[1:]#usuniecie peirwszej spolrzednej opowiadajacej wejsciu=1

        sygnal_delta1=dl_da1*fi_derative(W1X1)#[x*y for x in dl_da1 for y in fi_derative(W1X1) ][::(len(dl_da1)+1)]

        dl_W1=np.dot(sygnal_delta1,np.transpose(X1_train[i]))

        #zmiana W1
        W1_=W1-c*dl_W1

        W1=W1_
        W2=W2_
        W3=W3_

#print(W1,'\n\n\n',W2,'\n\n\n',W3)

ile_trafilismy=0
for i in range (0,10000):#10000-val_data
    #pierwsza warstwa
    W1X1=np.dot(W1,X1_val[i])
    a1=fi(W1X1)
    X2=np.vstack(( np.array([[1]]),a1))
    
    #druga warstwa
    W2X2=np.dot(W2,X2)
    a2=fi(W2X2)
    X3=np.vstack(( np.array([[1]]),a2))

    #trzecia warstwa
    W3X3=np.dot(W3,X3)
    a3=fi(W3X3)
    #print('a3: ',np.argmax(a3),'\n\n\n y:',Y_val[i])
    if np.argmax(a3)==Y_val[i]: ile_trafilismy+=1

#prawdopodobienstwo(trafienia)
print('prawdopodobienstwo(trafienia) = ',ile_trafilismy/10000)

"""
????? co znaczy w opisie zadania
zwraca błędy dla następnej warstwy; dobrze jakoś pamiętać obliczenia wykonane w czasie propagacji w przód). W prostszej (ale wystarczającej) wersji, możemy dawać Warstwie pojedynczy wektor wejściowy, wtedy da się naturalnie zaimplementować 'online learning'. Dla implementacji 'batch learning' lepiej, żeby Warstwa mogła dostać wektor wektorów wejściowych, czyli macierz


"""




"""
((a1,a2),(b1,b2),(c1,c2))=load_data() # a1 = 28x28 image = x , a2 = 1x10 = y 
(a,b,c)=load_data_wrapper()#zipy\
training_data_image, training_data_y = map(list, zip(*list(a))) 
#training_data=list(zip(*a))
#print(type(training_data[2]))
validation_data_image, validation_data_y= map(list, zip(*list(b)))
test_image, test_data_y= map(list, zip(*list(c)))
#training_data_image=training_data_image[0]
# dla upewnienia sie mozemy zobaczyc co otrzymalismy
#print(training_data_image[1])
#print(training_data_image[0])
#print('\n\n\n\n\n\n\n',validation_data_y)
#print('\n\n\n\n\n\n\n',test_data_y[1])
#print('\n\n\n\n\n\n\n',test_image[1])  
X1= [np.vstack((np.array([[1]]), lst)) for lst in training_data_image]# dodajemy jedynki na poczatek kazdej listy w liscie list
#print(type(training_data_image))
#print(X1[0]) # dodajemy jedynki na poczatek kazdej listy w liscie list
print(type(X1))

#X1 jest postaci [ [[1],[0],...,[0.9776]],  ...  ,[[1],...,[0]]    ]
#chcemy X1 postaci [ [1,0,...,0.9776], ..., [1,...,0]   ]

test=10 # dla X1 zbyt duzo czasu zajmuje wiec zrobmy dla pierwszych 10 list z X1
X1_test=[[0]]*test

# ?? BYT DLUGO SIE DLA X1
for j in range (0,test): 
    for i in range(0,784):
        X1[j][i]=X1[j][i][0]

for j in range (0,test):
    X1_test[j]=X1[j]

#WARSTWA 1
W1=np.random.normal(loc=0,scale=0.25,size=(784,785))
W1X1_test=[np.dot(W1,lst) for lst in X1_test]
#print(len(WX1_test[0]))
fi_W1X1_test = [fi(x) for x in W1X1_test]
print(fi_W1X1_test[0])

#WARSTWA 2
W2=np.random.normal(loc=0,scale=0.25,size=(784,785))
W2X1_test=[np.dot(W2,lst) for lst in X1_test]
#print(len(WX1_test[0]))
fi_W2X1_test = [fi(x) for x in W2X1_test]
print(fi_W2X1_test[0])
"""
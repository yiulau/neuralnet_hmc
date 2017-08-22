import cPickle, gzip, numpy, tarfile

#
path = '/home/yiulau/data/mnist.pkl.gz'
f = gzip.open(path,'rb')
out = cPickle.load(f)
f.close()

train_set,valid_set,test_set = out

import theano
import theano.tensor as T
import time

X=T.fmatrix()
Y=T.fmatrix()


w_h = theano.shared(numpy.asarray(numpy.random.randn(784,500), dtype=theano.config.floatX))
w_o = theano.shared(numpy.asarray(numpy.random.randn(500,10),dtype=theano.config.floatX))
h = T.nnet.sigmoid(T.dot(X,w_h))
py_x = T.nnet.softmax(T.dot(h,w_o))
y_x = T.argmax(py_x,axis=1)
cost = T.mean(T.nnet.categorical_crossentropy(py_x,Y))
params = [w_h, w_o]

def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

updates = sgd(cost, params)


train = theano.function(inputs=[X,Y],outputs=cost,updates=updates,allow_input_downcast=True)
predict  = theano.function(inputs=[X],outputs=y_x,allow_input_downcast=True)
#print(train_set[1][0:2].shape)
transformed_trainY = numpy.eye(10)[train_set[1]]
#print(transformed_trainY[0:2])
#one = numpy.eye(10)[train_set[1][0:2]]
#print(sampley)
#s = train(train_set[0][0:2],sampley)
#print(s)



for i in range(20):
    start_t = time.time()
    for start, end in zip(range(0, 50000,200), range(200, 50000, 200)):
        cost = train(train_set[0][start:end], transformed_trainY[start:end])
    total = time.time() - start_t
    print(numpy.mean(train_set[1] == predict(train_set[0])))
    print(total)
#s = train(train_set[0][0:120],transformed_trainY[0:120])
#print(s)
#t = predict(train_set[0][0:5])
#print(t)
#print(train_set[1][0:5])
#print(numpy.mean(train_set[1][0:5]==t))

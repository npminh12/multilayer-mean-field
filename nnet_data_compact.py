import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from skimage.transform import resize as image_resize

class nnet_data(object):
    def __init__(self, params):
        self.data_choice = params['data_choice']
        self.struct = params['data_structure']
        self.dataDim = params['d']
        self.format = params['format']
        self.range_normalize = params['range_normalize']
        
        if params['Numpy_randomSeed'] is not None:
            np.random.seed(params['Numpy_randomSeed'])                 
        
        self._generate_struct()
            
    def generate_struct(self):        
        self._generate_struct()
        
    def get_data(self, numData, whichSet='train'):
        X, y = self._generate_data(numData, whichSet=whichSet)
        X, y = self._change_format(X, y, _format=self.format)
        return X, y
    
    #---------------------------------------- Generate data
    def _generate_struct(self):        
        if self.data_choice=='mnist':            
            classes = self.struct['classes']
            labels = self.struct['labels']
            numTrain = self.struct['number_training_samples']
            X_train, X_val, X_test, y_train, y_val, y_test, dataDim = \
                self._mnist_import(numTrain=numTrain, range_normalize=self.range_normalize, 
                                     is_one_hot=False)
            self.X_train, self.y_train = self._get_classes(X_train, y_train, classes, labels)
            self.X_val, self.y_val = self._get_classes(X_val, y_val, classes, labels)
            self.X_test, self.y_test = self._get_classes(X_test, y_test, classes, labels)  
            del X_train, X_val, X_test
        elif self.data_choice=='cifar10':            
            classes = self.struct['classes']
            labels = self.struct['labels']
            numTrain = self.struct['number_training_samples']
            X_train, X_val, X_test, y_train, y_val, y_test, numTrain, numVal, numTest, dataDim = \
                self._cifar10_import(numTrain=numTrain, range_normalize=self.range_normalize)
            self.X_train, self.y_train = self._get_classes(X_train, y_train, classes, labels)
            self.X_val, self.y_val = self._get_classes(X_val, y_val, classes, labels)
            self.X_test, self.y_test = self._get_classes(X_test, y_test, classes, labels)    
            del X_train, X_val, X_test
        elif self.data_choice=='cifar10_feat':
            classes = self.struct['classes']
            labels = self.struct['labels']
            numTrain = self.struct['number_training_samples']
            input_shape = self.struct['new_input_shape']
            X_train, X_val, X_test, y_train, y_val, y_test, numTrain, numVal, numTest, dataDim = \
                self._cifar10_feat_import(numTrain=numTrain, input_shape=input_shape)
            self.X_train, self.y_train = self._get_classes(X_train, y_train, classes, labels)
            self.X_val, self.y_val = self._get_classes(X_val, y_val, classes, labels)
            self.X_test, self.y_test = self._get_classes(X_test, y_test, classes, labels) 
            del X_train, X_val, X_test
            
    
    def _generate_data(self, numData, whichSet='train'):       
        if self.data_choice=='isotropic Gaussian':
            labels0, labels1 = np.min(self.struct['labels']), np.max(self.struct['labels'])
            X = np.random.normal(scale=1.0, size=([numData] + self.dataDim))            
            sample = np.random.randint(low=0, high=2, size=numData)
            y = np.reshape(sample*labels1 + (1-sample)*labels0, newshape=(numData,1))
            sc = np.array([self.struct['stdin_low']*(1-sample[i]) + self.struct['stdin_high']*sample[i] \
                          for i in range(numData)])
            X = (X.T*sc).T
        elif self.data_choice in ['mnist', 'cifar10', 'cifar10_feat']:
            if whichSet=='train':
                num = self.X_train.shape[0]
                X = self.X_train
                y = self.y_train
            elif whichSet=='val':
                num = self.X_val.shape[0]
                X = self.X_val
                y = self.y_val
            elif whichSet=='test':
                num = self.X_test.shape[0]
                X = self.X_test
                y = self.y_test                
            if numData > num:
                numData = num                
            ind = np.random.choice(range(num), size=numData, replace=False)                        
            X = X[ind]
            y = np.reshape(y[ind], (numData, 1))            
        return X, y
    
    
    #---------------------------------------- Real data sets  
    def _mnist_import(self, numTrain=50000, range_normalize=None,\
                    is_one_hot=False, is_flatten=True):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=is_one_hot)
        train_data = mnist.train.images
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        # Reshape
        train_data = np.reshape(train_data, [train_data.shape[0], 28, 28, 1])
        eval_data = np.reshape(eval_data, [eval_data.shape[0], 28, 28, 1])
        
        # Make X_train, X_val, X_test
        numVal = train_data.shape[0] - numTrain
        ind = np.random.permutation(train_data.shape[0])
        X_train = train_data[ind[0:numTrain]]
        y_train = train_labels[ind[0:numTrain]]
        X_val = train_data[ind[numTrain:]]
        y_val = train_labels[ind[numTrain:]]        
        
        numTest = eval_data.shape[0]
        X_test = np.array(eval_data, copy=True)
        y_test = np.array(eval_labels, copy=True)
        
        dataDim = X_train.shape[1:]
                
        # range normalize
        X_train = self._do_range_normalize(X_train, range_normalize, [0.0, 1.0])
        X_val = self._do_range_normalize(X_val, range_normalize, [0.0, 1.0])
        X_test = self._do_range_normalize(X_test, range_normalize, [0.0, 1.0])
        
        # shuffle
        X_train, y_train = self._shuffle(X_train, y_train)
        X_val, y_val = self._shuffle(X_val, y_val)
        X_test, y_test = self._shuffle(X_test, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, dataDim

    
    def _cifar10_import(self, numTrain=45000, range_normalize=None):
        (train_data, train_labels), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        train_data = np.array(train_data)/255.0
        train_labels = np.array(train_labels)
        X_test = np.array(X_test)/255.0
        y_test = np.array(y_test)                            
        
        # Make X_train, X_val
        numVal = max(train_data.shape[0] - numTrain, 0)
        ind = np.random.permutation(train_data.shape[0])
        train_data = train_data[ind]
        train_labels = train_labels[ind]        
        X_val = train_data[numTrain:]
        X_train = train_data[0:numTrain]
        y_val = train_labels[numTrain:]
        y_train = train_labels[0:numTrain]
                
        # normalize
        X_train = self._do_range_normalize(X_train, range_normalize, [0.0, 1.0])
        X_val = self._do_range_normalize(X_val, range_normalize, [0.0, 1.0])
        X_test = self._do_range_normalize(X_test, range_normalize, [0.0, 1.0])
        
        numTrain = X_train.shape[0]
        numVal = X_val.shape[0]
        numTest = X_test.shape[0]
        dataDim = X_train.shape[1:]                        
        
        # shuffle        
        X_train, y_train = self._shuffle(X_train, y_train)
        X_val, y_val = self._shuffle(X_val, y_val)
        X_test, y_test = self._shuffle(X_test, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, numTrain, numVal, numTest, dataDim        

    
    def _cifar10_feat_import(self, numTrain=45000, input_shape=None):
        (train_data, train_labels), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()        
        train_labels = np.array(train_labels)
        y_test = np.array(y_test)                               
            
        # Extract features from the model                       
        if input_shape==None:
            input_shape = train_data[0].shape
            flag = False
        else:
            flag = True                
        model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        train_data = list(train_data)
        for i in range(len(train_data)):
            if flag:
                train_data[i] = image_resize(train_data[i], input_shape, preserve_range=True)
            train_data[i] = np.expand_dims(train_data[i], axis=0)
            train_data[i] = preprocess_input(train_data[i])
            train_data[i] = model.predict(train_data[i])
            train_data[i] = train_data[i][0]
        train_data = np.array(train_data, dtype=np.float32)
        X_test = list(X_test)
        for i in range(len(X_test)):
            if flag:
                X_test[i] = image_resize(X_test[i], input_shape, preserve_range=True)
            X_test[i] = np.expand_dims(X_test[i], axis=0)
            X_test[i] = preprocess_input(X_test[i])
            X_test[i] = model.predict(X_test[i])
            X_test[i] = X_test[i][0]
        X_test = np.array(X_test, dtype=np.float32)
            
        tf.keras.backend.clear_session()
        del model            
        
        # Make X_train, X_val
        numVal = max(train_data.shape[0] - numTrain, 0)
        ind = np.random.permutation(train_data.shape[0])
        train_data = train_data[ind]
        train_labels = train_labels[ind]        
        X_val = train_data[numTrain:]
        X_train = train_data[0:numTrain]
        y_val = train_labels[numTrain:]
        y_train = train_labels[0:numTrain]
                
        numTrain = X_train.shape[0]
        numVal = X_val.shape[0]
        numTest = X_test.shape[0]
        dataDim = X_train.shape[1:]                        
        
        # shuffle        
        X_train, y_train = self._shuffle(X_train, y_train)
        X_val, y_val = self._shuffle(X_val, y_val)
        X_test, y_test = self._shuffle(X_test, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, numTrain, numVal, numTest, dataDim        
    
    
    #---------------------------------------- Misc
    def _change_format(self, X, y, _format='column features'):
        if _format=='row features':
            return X, y
        elif _format=='column features':
            return X.T, y.T            

    def _get_classes(self, X, y, classes, labels):        
        if len(labels) != len(classes):
            raise NameError('Error: number of classes differs from number of labels!!!')
        numClass = len(classes)
        for cnt in range(numClass):
            ind = [i for i in range(len(y)) if y[i] in classes[cnt]]
            if cnt==0:
                X_new = X[ind]
                y_new = np.full(len(ind), labels[0], dtype=labels[0].dtype)
            else:
                X_new = np.append(X_new, X[ind], axis=0)
                y_new = np.append(y_new, np.full(len(ind), labels[cnt], dtype=labels[cnt].dtype))
        X_new, y_new = self._shuffle(X_new, y_new)        
        return X_new, y_new
    
    def _shuffle(self, X, y): 
        if X.shape[0]>0:
            ind = np.random.permutation(range(X.shape[0]))
            return X[ind], y[ind]
        else:
            return X, y

    def _do_range_normalize(self, X, range_normalize=None, original_range=None):    
        if range_normalize is not None:
            a, b = np.min(range_normalize), np.max(range_normalize)
            if original_range==None:                
                c, d = np.min(X), np.max(X)
            else:
                c, d = np.min(original_range), np.max(original_range)
            m = (b-a)*1.0/(d-c)        
            p = b - m*d
            X = X*m + p
        return X    
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
import math
from nnet_data_compact import *

#-----------------------------------------------------------------------------------------------------
# Neural net
#        
class nnet(object):
    def __init__(self, NN_architecture):
        self.dataDim = NN_architecture['dataDim']
        self.loss_choice = NN_architecture['loss_choice']
        self.labels = np.array(NN_architecture['labels'])
        self.transfer_func = NN_architecture['transfer_func']
        self.layer_type = NN_architecture['layer_type']
        self.layer_dim = NN_architecture['layer_dim']
        self.conv_strides = NN_architecture['conv_strides']
        self.pool_dim = NN_architecture['pool_dim']
        self.pool_strides = NN_architecture['pool_strides']
        self.pool_type = NN_architecture['pool_type']
        self.padding_type = NN_architecture['padding_type']
        self.W_init = NN_architecture['W_init']
        self.bias_init = NN_architecture['bias_init']   
        GPU_memory_fraction = NN_architecture['GPU_memory_fraction']
        GPU_which = NN_architecture['GPU_which']
        Tensorflow_randomSeed = NN_architecture['Tensorflow_randomSeed']
        self.dtype = NN_architecture['dtype']        
        
        self.depth = len(self.layer_dim)
                
        # check if valid
        self._is_params_valid()
        
        # reset graph     
        tf.reset_default_graph()
        if Tensorflow_randomSeed is not None:
            tf.set_random_seed(Tensorflow_randomSeed)
                
        # create graph, either using CPU or 1 GPU
        if GPU_which is not None:
            with tf.device('/device:GPU:' + str(GPU_which)):
                self._create_graph()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_memory_fraction)    
            config = tf.ConfigProto(gpu_options = gpu_options,\
                                    allow_soft_placement=True, log_device_placement=True)
        else:
            self._create_graph()            
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, \
                                device_count = {'GPU': 0})
        
        # initialize tf variables & launch the session
        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        
        # to save model
        self.saver = tf.train.Saver()
    
    #--------------------------------------------------------------------------
    # Check validity
    #        
    def _is_params_valid(self):
        for layer in range(2, self.depth):
            if self._layer_type(layer)=='conv' and self._layer_type(layer-1)=='fc':
                raise ValueError('Invalid params: all conv layers first, then all fc layers!')
                
        if self.loss_choice=='cross entropy':
            if len(self.labels)!=self._output_size():
                raise ValueError('Invalid params: with logit loss, output dimension = number of classes!')
            if self.labels.dtype!=np.int32:
                raise ValueError('Invalid params: with logit loss, labels must be np.int32!')
            if np.setdiff1d(self.labels, np.array(range(len(self.labels)), dtype=np.int32)).size>0:
                raise ValueError('Invalid params: with logit loss, labels must be [0,1,...,number of classes-1]!')
            
        if self.loss_choice=='squared':
            if self._output_size()>1:
                raise ValueError('Invalid params: with squared loss, output dimension = 1!')
            if self.labels.dtype!=np.float32:
                raise ValueError('Invalid params: with squared, labels must be np.float32!')
        
    
    #--------------------------------------------------------------------------
    # Create the graph
    #        
    def _create_graph(self):
        self.x = tf.placeholder(self.dtype, [None, self.dataDim[0], self.dataDim[1], self.dataDim[2]])  
        self.y = self._label_placeholder()       
        self.learning_rate = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        
        self._create_nnet()
        self._create_loss_optimizer()                
    
    def _label_placeholder(self):
        if self.loss_choice=='squared':
            return tf.placeholder(tf.float32, [None, 1])
        elif self.loss_choice=='cross entropy':
            return tf.placeholder(tf.int32, [None, 1])
    
    #--------------------------------------------------------------------------
    # Create the neural net graph
    #    
    def _create_nnet(self):        
        x = self.x
        W = []
        b = []
        preact = []
        for layer in range(1,self.depth+1):            
            # Reshape if needed
            if (layer==1 and self._layer_type(layer)=='fc') or \
                    (layer>1 and self._layer_type(layer)=='fc' and self._layer_type(layer-1)=='conv'):
                _, _, channel_in, _, _ = self._layer_dim(layer)
                x = tf.reshape(x, (-1, channel_in))
            
            # Applying weight
            W.append(self._make_weight(layer))
            b.append(tf.Variable(self._bias_init(layer)))
            
            if layer>1:                
                _, _, _, _, dim = self._layer_dim(layer)
                mul = 1.0/dim
            else:
                mul = 1.0  
            if self._layer_type(layer)=='conv':                    
                x = tf.nn.conv2d(x, W[layer-1], strides=self._conv_strides(layer), padding=self.padding_type)*mul + b[layer-1]
            else:
                x = tf.matmul(x, W[layer-1])*mul + b[layer-1]                    
            preact.append(x)
            
            # Nonlinearity
            if layer < self.depth:
                x = self._transfer(x, layer)           
            
            # Pooling
            if self._layer_type(layer)=='conv' and self._is_pool(layer):
                x = tf.nn.pool(x, window_shape=self._pool_dim(layer), pooling_type=self._pool_type(layer),
                               strides=self._pool_strides(layer), padding=self.padding_type, data_format='NHWC')   
                
        self.yhat = self._reshape_output(x)        
        self.W = W
        self.b = b
        self.preact = preact
        
    def _reshape_output(self, yhat):
        if self.loss_choice=='squared':
            return tf.reshape(yhat, (-1, 1))
        elif self.loss_choice=='cross entropy':
            return yhat  
    
    #--------------------------------------------------------------------------
    # Return the dimension for the 'layer'-th layer
    #    
    def _layer_type(self, layer):
        return self.layer_type[layer-1]
    
    def _layer_dim(self, layer):    
        if self._layer_type(layer)=='conv':
            temp = self.layer_dim[layer-1]
            width, height, channel_out = temp[0], temp[1], temp[2]
            if layer==1:
                channel_in = self.dataDim[2]
            else:
                channel_in = self.layer_dim[layer-2][2]
            dim = channel_in*width*height
        else:
            if layer==1:
                channel_in = np.prod(self.dataDim)
            else:                
                channel_in = self.layer_dim[layer-1][0]
            channel_out = self.layer_dim[layer-1][1]
            width = None
            height = None
            dim = channel_in
        return width, height, channel_in, channel_out, dim   
    
    def _conv_strides(self, layer):
        temp = self.conv_strides[layer-1]
        height, width = temp[0], temp[1]
        return [1, height, width, 1]    
    
    def _pool_dim(self, layer):
        if self._is_pool(layer):
            temp = self.pool_dim[layer-1]
            height, width = temp[0], temp[1]
        else:
            height, width = 0, 0
        return [height, width]
    
    def _pool_strides(self, layer):
        if self._is_pool(layer):
            temp = self.pool_strides[layer-1]
            height, width = temp[0], temp[1]
        else:
            height, width = 0, 0
        return [height, width]
    
    def _is_pool(self, layer):
        return len(self.pool_dim[layer-1])>0
    
    def _output_size(self):
        _, _, _, channel_out, _ = self._layer_dim(self.depth)
        return channel_out
    
    def _pool_type(self, layer):
        if self.pool_type[layer-1]=='MAX':
            return 'MAX'
        if self.pool_type[layer-1]=='AVG':
            return 'AVG'        
        
    #--------------------------------------------------------------------------
    # Initialization of the weights, biases
    #
    def _weight_init(self, layer):
        height, width, channel_in, channel_out, dim = self._layer_dim(layer)
        if self._layer_type(layer)=='conv':                
            dims = [height, width, channel_in, channel_out]
        else:
            dims = [channel_in, channel_out]            
        if layer==1:
            std = self.W_init['params']['std'][layer-1]/np.sqrt(dim)
            mean = self.W_init['params']['mean'][layer-1]/dim
        else:
            std = self.W_init['params']['std'][layer-1]
            mean = self.W_init['params']['mean'][layer-1]
        gauss = tf.random_normal(dims, mean=mean, stddev=std, dtype=self.dtype)
        return gauss
                    
    def _make_weight(self, layer):
        return tf.Variable(self._weight_init(layer), trainable=True, dtype=self.dtype)
        
    def _bias_init(self, layer):
        _, _, _, channel_out, _ = self._layer_dim(layer)
        mean = self.bias_init['mean']
        std = self.bias_init['std']
        return tf.random_normal([channel_out], mean=mean, stddev=std, dtype=self.dtype)
    
    #--------------------------------------------------------------------------
    # Define the nonlinearity
    #
    def _transfer(self, x, layer):
        return self._nonlinearity(x, self.transfer_func[layer-1])    
    
    def _nonlinearity(self, x, transfer_func='relu'):
        if transfer_func=='relu':
            return tf.nn.relu(x)
        elif transfer_func=='tanh':
            return tf.tanh(x) 
        
        
    #--------------------------------------------------------------------------
    # Create the loss and the optimizer
    #
    def _create_loss_optimizer(self):
        self.loss = self._create_loss()       
        opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)        
        grads_and_vars = []            
        gv = opt.compute_gradients(self.loss, self.W)
        for i in range(1,self.depth+1):
            grads_and_vars.append((self._process_W_grad(gv[i-1][0], i), gv[i-1][1]))
        gv = opt.compute_gradients(self.loss, self.b)
        for i in range(1,self.depth+1):
            grads_and_vars.append((self._process_bias_grad(gv[i-1][0], i), gv[i-1][1]))            
        self.optimizer = opt.apply_gradients(grads_and_vars)             
        
    def _create_loss(self):
        if self.loss_choice=='squared':
            cost = tf.reduce_mean(tf.square(self.yhat - self.y))
        elif self.loss_choice=='cross entropy':
            labels = tf.squeeze(self.y)
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.yhat))  
        return cost
        
    def _process_W_grad(self, grad, layer):                        
        if layer==1:
            _, _, _, _, dim = self._layer_dim(layer+1)
            mul = dim
        elif layer<self.depth:
            _, _, _, _, dim1 = self._layer_dim(layer)
            _, _, _, _, dim2 = self._layer_dim(layer+1)
            mul = dim1*dim2
        else:
            _, _, _, channel_out, dim = self._layer_dim(layer)
            mul = dim*channel_out
        return grad*mul    
    
    def _process_bias_grad(self, grad, layer):
        if layer<self.depth:
            _, _, _, _, dim = self._layer_dim(layer+1)
            mul = dim
        else:
            mul = self._output_size()
        return grad*mul
    
    #--------------------------------------------------------------------------
    # Auxilliary functions
    #        
    def _compute_error(self, y, pred): 
        if self.loss_choice=='squared':
            y = np.squeeze(y)
            pred = np.squeeze(pred)
            newy = np.array([np.argmin(np.abs(self.labels-y[cnt])) for cnt in range(len(y))])
            newpred = np.array([np.argmin(np.abs(self.labels-pred[cnt])) for cnt in range(len(pred))])
            error = np.mean(newy!=newpred)
        elif self.loss_choice=='cross entropy':
            y = np.squeeze(y)
            pred = np.argmax(pred, axis=1)
            error = np.mean(y!=pred)
        return error    
    
    #--------------------------------------------------------------------------
    # Public methods
    #
    def fit(self, x, y, learning_rate):
        _, loss = self.sess.run((self.optimizer, self.loss), \
                                feed_dict={self.x: x, self.y: y, \
                                           self.learning_rate: learning_rate, 
                                           self.is_training: True})
        return loss
    
    def predict(self, x, y=None, batch_size=100):
        if y is None:
            ind = 0
            pred = np.zeros((x.shape[0],self._output_size()))
            while ind < x.shape[0]:
                temp = x[ind:(ind+batch_size)]
                temppred = self.sess.run(self.yhat, feed_dict={self.x: temp, self.is_training: False})
                pred[ind:(ind+batch_size)] = temppred
                ind += temp.shape[0]
            return pred
        else:            
            ind = 0
            pred = np.zeros((x.shape[0],self._output_size()))
            loss = 0
            while ind < x.shape[0]:
                temp = x[ind:(ind+batch_size)]
                tempy = y[ind:(ind+batch_size)]
                temppred, temp_loss = self.sess.run((self.yhat, self.loss), \
                                                    feed_dict={self.x: temp, self.y: tempy, self.is_training: False})
                pred[ind:(ind+batch_size)] = temppred
                ind += temp.shape[0]
                loss += temp_loss*temp.shape[0]
            loss /= x.shape[0]                             
            error = self._compute_error(y, pred)
            return pred, loss, error        
        
    def preact_thresh(self, x, thresh, batch_size=100):  
        # average number of preactivation entries > thresh, averaged over all inputs in x
        ind = 0
        preact_thresh = np.zeros(self.depth)
        preact_size = np.zeros(self.depth)
        while ind < x.shape[0]:
            temp = x[ind:(ind+batch_size)]            
            preact = self.sess.run((self.preact), feed_dict={self.x: temp, self.is_training: False})
            for cnt in range(self.depth):
                preact_thresh[cnt] += np.sum(preact[cnt]>thresh)
                preact_size[cnt] = np.prod(preact[cnt].shape[1:])
            ind += temp.shape[0]
        for cnt in range(self.depth):
            preact_thresh[cnt] /= x.shape[0]*preact_size[cnt]
        return preact_thresh    
        
    
#-----------------------------------------------------------------------------------------------------
# Simulation of the neural net
#        
class nnet_simul(object):
    def __init__(self, params):
        self.NN = params['neural_net']
        self.data = params['data']
        self.SGD = params['SGD']                                
        self.statsCollect = params['statsCollect']      
        
        self.depth = len(self.NN['layer_dim'])        
        self.stats = {}
        
    #--------------------------------------------------------------------------
    # Generate the data
    #    
    def _generate_data_module(self):
        self.data['format'] = 'row features'
        nn_data = nnet_data(self.data) 
        return nn_data
        
    def _generate_data(self, numData, whichSet='train'):
        return self.nnet_data.get_data(numData, whichSet=whichSet)
    
    #--------------------------------------------------------------------------
    # Create the neural net
    #    
    def _generate_nnet(self):
        NN_architecture = self.NN
        NN_architecture['dataDim'] = self.data['d']
        NN_architecture['labels'] = self.data['data_structure']['labels']
        return nnet(NN_architecture)
        
    #--------------------------------------------------------------------------
    # Collect statistics
    #    
    def _statistics(self, numMonteCarlo):
        X, y = self._generate_data(numMonteCarlo, whichSet='test')
        _, loss_test, error_test = self.nnet.predict(x=X, y=y)    
        X, y = self._generate_data(numMonteCarlo, whichSet='train')
        _, loss_train, error_train = self.nnet.predict(x=X, y=y)                
        stats = dict(
            loss_test = loss_test,           
            loss_train = loss_train,
            error_test = error_test, 
            error_train = error_train,
        )                                
        
        # additional stats
        addStats = self.statsCollect['additional_stats']
        for cnt in range(len(addStats)):
            if addStats[cnt]['which']=='preact > threshold' and addStats[cnt]['run']==True:
                name = addStats[cnt]['name']
                thresh = addStats[cnt]['params']['threshold']
                X, _ = self._generate_data(numMonteCarlo, whichSet='test')
                temp = self.nnet.preact_thresh(x=X, thresh=thresh)   
                stats[name] = {}
                for i in range(1,self.depth+1):
                    stats[name][i] = temp[i-1]
        
        return stats
    
    #--------------------------------------------------------------------------
    # Perform training with SGD
    #    
    def _SGD_update(self, iteration):                        
        # generate data
        x, y = self._generate_data(self.SGD['batchSize'], whichSet='train')
        
        # SGD learning rate        
        lr = self.SGD['stepSize']/(iteration**self.SGD['decay_power'])
          
        # update        
        self.nnet.fit(x, np.reshape(np.array(y, ndmin=1), (self.SGD['batchSize'], 1)), lr)
        return lr
            
    def _SGD_run(self, iter_start=1):
        num = self.statsCollect['numMonteCarlo']        
        
        # run SGD
        time = timer()    
        if self.statsCollect['is_verbose']:
            print('Iter | Time (min) | Learning rate | Train loss | Test loss | Train error | Test error')
        for iteration in range(iter_start, iter_start+self.SGD['iteration_num']):
            lr = self._SGD_update(iteration)            
            if iteration in self.statsCollect['output_schedule']:
                self._update_stats(iteration, num)
                
                time = timer() - time  
                if self.statsCollect['is_verbose']:
                    print('%08d' % (iteration), 
                          "|",  '%.3f' % (time/60),
                          "|", '%.3e' % (lr),
                          "|", '%.3e' % (self.stats[iteration]['loss_train']),
                          "|", '%.3e' % (self.stats[iteration]['loss_test']),
                          "|", '%.3e' % (self.stats[iteration]['error_train']),
                          "|", '%.3e' % (self.stats[iteration]['error_test'])
                         )
                time = timer()

    def _update_stats(self, iteration, numMonteCarlo):
        self.stats[iteration] = self._statistics(numMonteCarlo)
    
                
    #--------------------------------------------------------------------------
    # Public methods
    #        
    def generate_nnet(self):
        self.nnet = self._generate_nnet()
        self.nnet_data = self._generate_data_module()
    
    def run(self, iter_start=1):
        self.generate_nnet()
        self._SGD_run(iter_start)
    
    def collect_stats(self):
        return self.stats
    
    def get_nnet(self):
        return self.nnet
    
    def predict(self, X):
        return self.nnet.predict(X)    
    
    def get_depth(self):
        return len(self.NN['layer_dim'])
        
#-----------------------------------------------------------------------------------------------------
# Plot
#       
def save_pickle(FILENAME, nnet_simul_obj, params, elapsed_min):
    import pickle
    
    with open(FILENAME + ".pkl", 'wb') as myFILE:
        pickle.dump([nnet_simul_obj.collect_stats(), params, elapsed_min], myFILE)        
        
def load_pickle(FILENAME):
    import pickle
    
    myFILE = open(FILENAME + '.pkl', 'rb')
    stats, params, elapsed_min = pickle.load(myFILE)
    return stats, params, elapsed_min

def plot_results(obj_all, nbins=20, figSize=3, obj_labels=None, obj_style=None):
    import matplotlib.pyplot as plt     
    
    if obj_style==None:
        obj_style = ['b-', 'g-', 'r-', 'k-', 'm-', 'y-', 'bx--', 'gx--', 'rx--', 'kx--', 'yx--', 'mx--',\
                     'b.-', 'g.-', 'r.-', 'k.-', 'y.-', 'm.-', 'b*:', 'g*:', 'r*:', 'k*:', 'y*:', 'm*:']

    ## Get max depth
    depth_max = 0
    for cnt_obj in range(len(obj_all)):
        if depth_max < obj_all[cnt_obj].get_depth():
            depth_max = obj_all[cnt_obj].get_depth()
    
    ## Plot loss, error
    item_list = ['loss_train', 'loss_test', 'error_train', 'error_test']    
    fig, ax = plt.subplots(1,4, figsize=(figSize*4,figSize))
    count = 0
    for cnt in range(len(item_list)):
        what = item_list[cnt]    
        for cnt_obj in range(len(obj_all)):
            val = []
            std = []
            for i in obj_all[cnt_obj].statsCollect['output_schedule']:
                val.append(obj_all[cnt_obj].stats[i][what])
            ax[count].semilogx(obj_all[cnt_obj].statsCollect['output_schedule'], val, obj_style[cnt_obj%len(obj_style)])
        ax[count].grid(color='k', linestyle='--', linewidth=0.5) 
        ax[count].set_title(what)                  
        if obj_labels != None:
            ax[count].legend(obj_labels)
        count += 1
    plt.tight_layout()
    plt.show()
    
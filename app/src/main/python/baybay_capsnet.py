import tensorflow as tf
import numpy as np

(X_train, Y_train), (X_valid, Y_valid) = (None, None), (None, None)

#############################
#### PRE-PROCESSING FUNCTIONS

# normalize dataset
def pre_process(image, label):
    return (image / 256)[...,None].astype('float32'), tf.keras.utils.to_categorical(label, num_classes=63)
    
def pre_process_no_label(image):
    return (image / 256)[...,None].astype('float32')

########################
#### LAYERS & FUNCTIONS

class Squash(tf.keras.layers.Layer):
    """
    Squash activation used in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'.
    
    ...
    
    Attributes
    ----------
    eps: int
        fuzz factor used in numeric expression
    
    Methods
    -------
    call(s)
        compute the activation from input capsules
    """

    def __init__(self, eps=10e-21, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, s):
        n = tf.norm(s,axis=-1,keepdims=True)
        return (1 - 1/(tf.math.exp(n)+self.eps))*(s/(n+self.eps))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape



class Length(tf.keras.layers.Layer):
    """
    Compute the length of each capsule n of a layer l.
    ...
    
    Methods
    -------
    call(inputs)
        compute the length of each capsule
    """

    def call(self, inputs, **kwargs):
        """
        Compute the length of each capsule
        
        Parameters
        ----------
        inputs: tensor
           tensor with shape [None, num_capsules (N), dim_capsules (D)]
        """
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), - 1) + tf.keras.backend.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config



class Mask(tf.keras.layers.Layer):
    """
    Mask operation described in 'Dynamic routinig between capsules'.
    
    ...
    
    Methods
    -------
    call(inputs, double_mask)
        mask a capsule layer
        set double_mask for multimnist dataset
    """
    def call(self, inputs, double_mask=None, **kwargs):
        if type(inputs) is list:
            if double_mask:
                inputs, mask1, mask2 = inputs
            else:
                inputs, mask = inputs
        else:  
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            if double_mask:
                mask1 = tf.keras.backend.one_hot(tf.argsort(x,direction='DESCENDING',axis=-1)[...,0],num_classes=x.get_shape().as_list()[1])
                mask2 = tf.keras.backend.one_hot(tf.argsort(x,direction='DESCENDING',axis=-1)[...,1],num_classes=x.get_shape().as_list()[1])
            else:
                mask = tf.keras.backend.one_hot(indices=tf.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        if double_mask:
            masked1 = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask1, -1))
            masked2 = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask2, -1))
            return masked1, masked2
        else:
            masked = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask, -1))
            return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # generation step
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config

###########################
#### PRIMARY CAPSULE LAYER

class PrimaryCaps(tf.keras.layers.Layer):
    """
    Create a primary capsule layer with the methodology described in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'. 
    Properties of each capsule s_n are exatracted using a 2D depthwise convolution.
    
    ...
    
    Attributes
    ----------
    F: int
        depthwise conv number of features
    K: int
        depthwise conv kernel dimension
    N: int
        number of primary capsules
    D: int
        primary capsules dimension (number of properties)
    s: int
        depthwise conv strides
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """
    def __init__(self, F, K, N, D, s=1, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.F = F
        self.K = K
        self.N = N
        self.D = D
        self.s = s
        
    def build(self, input_shape):    
        self.DW_Conv2D = tf.keras.layers.Conv2D(self.F, self.K, self.s,
                                             activation='linear', groups=self.F, padding='valid')

        self.built = True
    
    def call(self, inputs):      
        x = self.DW_Conv2D(inputs)      
        x = tf.keras.layers.Reshape((self.N, self.D))(x)
        x = Squash()(x)
        
        return x
    

    def get_config(self):
        config = {
            'F': self.F,
            'K': self.K,
            'N': self.N,
            'D': self.D,
            's': self.s
        }
        base_config = super(PrimaryCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

##################################
#### FULLY-CONNECTED CAPSULE LAYER

class FCCaps(tf.keras.layers.Layer):
    """
    Fully-connected caps layer. It exploites the routing mechanism, explained in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing', 
    to create a parent layer of capsules. 
    
    ...
    
    Attributes
    ----------
    N: int
        number of primary capsules
    D: int
        primary capsules dimension (number of properties)
    kernel_initilizer: str
        matrix W initialization strategy
 
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """
    def __init__(self, N, D, kernel_initializer='he_normal', **kwargs):
        super(FCCaps, self).__init__(**kwargs)
        self.N = N
        self.D = D
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        input_N = input_shape[-2]
        input_D = input_shape[-1]

        self.W = self.add_weight(shape=[self.N, input_N, input_D, self.D],initializer=self.kernel_initializer,name='W')
        self.b = self.add_weight(shape=[self.N, input_N,1], initializer=tf.zeros_initializer(), name='b')
        self.built = True
    
    def call(self, inputs, training=None):
        
        u = tf.einsum('...ji,kjiz->...kjz',inputs,self.W)    # u shape=(None,N,H*W*input_N,D)
             
        c = tf.einsum('...ij,...kj->...i', u, u)[...,None]        # b shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        c = c/tf.sqrt(tf.cast(self.D, tf.float32))
        c = tf.nn.softmax(c, axis=1)                             # c shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        c = c + self.b
        s = tf.reduce_sum(tf.multiply(u, c),axis=-2)             # s shape=(None,N,D)
        v = Squash()(s)       # v shape=(None,N,D)
        
        return v

    def compute_output_shape(self, input_shape):
        return (None, self.C, self.L)

    def get_config(self):
        config = {
            'N': self.N,
            'D': self.D
        }
        base_config = super(FCCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#########################################
#### CASPNET ARCHITECTURE (EfficientCaps)

def efficient_capsnet_graph(input_shape):
    """
    Efficient-CapsNet graph architecture.

    Parameters
    ----------   
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(input_shape)
    
    x = tf.keras.layers.Conv2D(32,5,activation="relu", padding='valid', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64,3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64,3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)   
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128,3,2, activation='relu', padding='valid', kernel_initializer='he_normal')(x)   
    x = tf.keras.layers.BatchNormalization()(x)
    x = PrimaryCaps(128, 9, 16, 8)(x)
    
    baybayin_caps = FCCaps(63,16)(x)
    
    baybayin_caps_len = Length(name='length_capsnet_output')(baybayin_caps)

    return tf.keras.Model(inputs=inputs,outputs=[baybayin_caps, baybayin_caps_len], name='Efficient_CapsNet')


def generator_graph(input_shape):
    """
    Generator graph architecture.

    Parameters
    ----------   
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(16*63)
    
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid', kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Reshape(target_shape=input_shape, name='out_generator')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def build_graph(input_shape, verbose):
    """
    Efficient-CapsNet graph architecture with reconstruction regularizer. The network can be initialized with different modalities.

    Parameters
    ----------   
    input_shape: list
        network input shape
    verbose: bool
    """
    inputs = tf.keras.Input(input_shape)

    efficient_capsnet = efficient_capsnet_graph(input_shape)

    if verbose:
        efficient_capsnet.summary()
        print("\n\n")
    
    baybayin_caps, baybayin_caps_len = efficient_capsnet(inputs)
    
    masked = Mask()(baybayin_caps)
    
    generator = generator_graph(input_shape)

    if verbose:
        generator.summary()
        print("\n\n")

    x_gen_eval = generator(masked)

    return tf.keras.models.Model(inputs, [baybayin_caps_len, x_gen_eval], name='Efficient_CapsNet_Generator')

######################
#### MODEL DEFINITIONS

class Model(object):
    """
    A class used to share common model functions and attributes.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    config_path: str
        path configuration file
    verbose: bool
    
    Methods
    -------
    load_graph_weights():
        load network weights
    predict(dataset_test):
        use the model to predict dataset_test
    evaluate(X_test, y_test):
        compute accuracy and test error with the given dataset (X_test, y_test)
    save_graph_weights():
        save model weights
    """
    def __init__(self, model_name, verbose=True):
        self.model_name = model_name
        self.model = None
        self.verbose = verbose
    

    def load_graph_weights(self):
        try:
            self.model.load_weights(self.model_path)
        except Exception as e:
            print("[ERRROR] Graph Weights not found")
            

    def save_graph_weights(self):
        self.model.save_weights(self.model_path)

    def predict(self, dataset_test):
        return self.model.predict(dataset_test)

    def evaluate(self, X_test, y_test):
        print('-'*30 + f'{self.model_name} Evaluation' + '-'*30)

        y_pred, X_gen =  self.model.predict(X_test)
        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]

        test_error = 1 - acc
        print('Test acc:', acc*100)
        print(f"Test error [%]: {(test_error):.4%}")

        print(f"N° misclassified images: {int(test_error*len(y_test))} out of {len(y_test)}")



class EfficientCapsNet(Model):
    """
    A class used to manage an Efficiet-CapsNet model. 'model_name' define the particular architecure and modality of the 
    generated network.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    config_path: str
        path configuration file
    custom_path: str
        custom weights path
    verbose: bool
    
    Methods
    -------
    load_graph():
        load the network graph given the model_name
    train(dataset, initial_epoch)
        train the constructed network with a given dataset. All train hyperparameters are defined in the configuration file

    """
    def __init__(self, model_name, custom_path=None, verbose=True):
        Model.__init__(self, model_name, verbose)
        self.model_path = custom_path
        self.load_graph()
    

    def load_graph(self):
        self.model = build_graph([28,28,1], self.verbose)
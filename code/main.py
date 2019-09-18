
import subprocess
# subprocess.call(["bash","download_files.sh"])
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, numpy, string
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import layers, models, optimizers
import tensorflow as tf

try:
    import foundations
except Exception as e:
    print(e)



# In[6]:


print(tf.__version__)


# In[8]:
os.listdir('/data')

trainDF = pd.read_csv('tweet_labels.csv')

trainDF.head()


# In[10]:

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['tweet'], trainDF['labels'])

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.transform(valid_y)


# In[12]:


tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['tweet'])


def random_select_from_list(valid_value_list):
    random_ind = np.random.randint(len(valid_value_list))
    return valid_value_list[random_ind]

def generate_config():
    model_params = {'spatial_dropout': [0, 0.1,0.2,0.4,0.5],
                         'num_conv_blocks': [1,2,3],
                         'num_conv_filters': [10,30,50,70],
                         'filter_size': [3,5,7],
                         'num_dense_layers': [1,2],
                         'num_dense_neurons': [10,50,80],
                         'activation_func': ['relu', 'elu', 'selu', 'tanh'],
                         'dense_dropout': [0, 0.2, 0.5],
                         'epochs': [10],
                         'batch_size':[10, 20],
                         'learning_rate': [0.1, 0.01, 0.001, 0.0001, 0.00001]
                         }

    for k,v in model_params.items():
        model_params[k] = random_select_from_list(v)
    return model_params

model_params_conv = generate_config()

try:
    for k,v in model_params_conv.items():
        foundations.log_param(k, v)
except Exception as e:
    print(e)

# In[13]:

def save_plot(inp_list, fig_name):
    plt.figure()
    plt.plot(inp_list)
    plt.ylabel(f'{fig_name}')
    plt.xlabel('epochs')
    plt.title(f'{fig_name}')
    plt.savefig(f'{fig_name}.png')


def save_plot_all(inp_list, fig_name):
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ct=0
    for row in ax:
        for col in row:
            col.plot(inp_list[ct])
            col.set_ylabel(f'{fig_name[ct]}')
            col.set_xlabel('epochs')
            col.set_title(f'{fig_name[ct]}')
            ct+=1
    plt.savefig(f'performance_plots.png')

def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y,is_neural_net=False, params={}):
    # fit the training dataset on the classifier
    if is_neural_net:
        history = classifier.fit(feature_vector_train, label,
                       epochs=model_params_conv['epochs'],
                       validation_split=0.2,
                       batch_size = model_params_conv['batch_size'])

        history_dict = history.history
        train_loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        train_acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        inp_list = [train_loss, val_loss, train_acc, val_acc]
        fig_name = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        save_plot_all(inp_list, fig_name)

        try:
            foundations.save_artifact('performance_plots.png', key='performance_plots')
        except Exception as e:
            print(e)


    else:
        classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        #round probabilities to 1 and 0
        predictions = predictions.round()

        return metrics.accuracy_score(predictions, valid_y)
    return metrics.accuracy_score(predictions, valid_y)


# In[14]:


embeddings_index = {}
try:
    for i, line in enumerate(open('../data/wiki-news-300d-1M.vec')):
        values = line.split()
        embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
except:
    # When running with foundations, the data will be mounted on the root/data/ directory
    for i, line in enumerate(open('/data/wiki-news-300d-1M.vec')):
        values = line.split()
        embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')


# create a tokenizer
token = text.Tokenizer()
token.fit_on_texts(trainDF['tweet'])
word_index = token.word_index


# create token-embedding mapping
print("recalculating embedding matrix")
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# convert text to sequence of tokens and pad them to ensure equal length vectors
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# Just for speed we only take first 500 tweets
train_seq_x = train_seq_x[:500]
valid_seq_x = valid_seq_x[:500]
train_y = train_y[:500]
valid_y = valid_y[:500]




# In[15]:


def create_cnn(model_params):
    # Add an Input Layer, expected vectors of 0 and 1, with one vector for each word
    input_layer = layers.Input((70,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=True)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(model_params['spatial_dropout'])(embedding_layer)

    # Add the convolutional Layer
    for ly in range(model_params['num_conv_blocks']):
        if ly==0:
            conv_layer = layers.Convolution1D(model_params['num_conv_filters'], model_params['filter_size'], activation=model_params['activation_func'])(embedding_layer)
        else:
            conv_layer = layers.Convolution1D(model_params['num_conv_filters']*ly*2, model_params['filter_size'],
                                              activation=model_params['activation_func'])(conv_layer)
    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    for ly in range(model_params['num_dense_layers']):
        if ly==0:
            output_layer1 = layers.Dense(model_params['num_dense_neurons'], activation=model_params['activation_func'])(pooling_layer)
            output_layer1 = layers.Dropout(model_params['dense_dropout'])(output_layer1)
        else:
            output_layer1 = layers.Dense(model_params['num_dense_neurons'], activation=model_params['activation_func'])(output_layer1)
            output_layer1 = layers.Dropout(model_params['dense_dropout'])(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(lr=model_params['learning_rate'], decay=0.0001 / 30), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


classifier = create_cnn(model_params_conv)
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y,is_neural_net=True)
print("CNN, Word Embeddings",  accuracy)

try:
    foundations.log_metric('val_accuracy', accuracy)
except Exception as e:
    print(e)

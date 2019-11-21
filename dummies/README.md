import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

!pip install pytorch-pretrained-bert pytorch-nlp



import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

df.shape

df.sample(10)

sentences = df.sentence.values

# We need to add special tokens at the beginning and end of each sentence for BERT to work properly
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print ("Tokenize the first sentence:")
print (tokenized_texts[0])


### tokenize text looks like this 

'.',
  '[SEP]'],
 ['[CLS]',
  'the',
  'moral',
  'destruction',
  'of',
  'the',
  'president',
  'was',
  'certainly',
  'not',
  'helpful',
  '.',
  '[SEP]'],
 ['[CLS]',
  'mary',
  'wants',
  'to',
  'wear',
  'nice',
  'blue',
  'german',
  'dress',



# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway. 
# In the original paper, the authors used a length of 512.
MAX_LEN = 128

# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

array([[ 101, 2256, 2814, ...,    0,    0,    0],
       [ 101, 2028, 2062, ...,    0,    0,    0],
       [ 101, 2028, 2062, ...,    0,    0,    0],
       ...,
       [ 101, 2009, 2003, ...,    0,    0,    0],
       [ 101, 1045, 2018, ...,    0,    0,    0],
       [ 101, 2054, 2035, ...,    0,    0,    0]])


##############################################
# Create attention masks
attention_masks = []
##############################################

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)



# Use train_test_split to split our data into train and validation sets for training

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)


train_inputs looks like this 

tensor([[  101,  2002,  2939,  ...,     0,     0,     0],
        [  101,  1998,  2009,  ...,     0,     0,     0],
        [  101, 15609,  2038,  ...,     0,     0,     0],
        ...,
        [  101,  2048,  3719,  ...,     0,     0,     0],
        [  101,  2348, 23848,  ...,     0,     0,     0],
        [  101, 13723,  8487,  ...,     0,     0,     0]])

train_labels
tensor([1, 1, 1,  ..., 1, 0, 1])


validation_inputs
tensor([[  101,  1996, 13176,  ...,     0,     0,     0],
        [  101,  2057,  6955,  ...,     0,     0,     0],
        [  101,  2984,  7659,  ...,     0,     0,     0],
        ...,
        [  101,  2151,  5205,  ...,     0,     0,     0],
        [  101,  2002,  1005,  ...,     0,     0,     0],
        [  101, 22725,  2626,  ...,     0,     0,     0]])

    


# Convert all of our data into torch tensors, the required datatype for our model

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)



### Best advise ###

# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batch_size = 32

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

###
### STOP 
### From this point onwards we are using BertForSequenceClassification 
### https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification

## Bert is good for a lot of things, but for this task, it is classification

From input above, feed it into our model and the output is the length 768 hidden state vector corresponding to this token. The additional layer that we've added on top consists of untrained linear neurons of size [hidden_state, number_of_labels], so [768,2], meaning that the output of BERT plus our classification layer is a vector of two numbers representing the "score" for "grammatical/non-grammatical" that are then fed into cross-entropy loss. 




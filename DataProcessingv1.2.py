import json
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
from SentimentLSTM import SentimentLSTM
import torch.nn as nn
# %matplotlib inline

dirpath = '/geode2/home/u010/bagrawal/Carbonate/SearchFinalProject/dataset'

# review = []
# for line in open(dirpath + '/review.json', 'r'):
#     review.append(json.loads(line))

# tip = []
# for line in open(dirpath + '/tip.json', 'r'):
#     tip.append(json.loads(line))

# review_df = pd.DataFrame.from_dict(review)

# tip_df = pd.DataFrame.from_dict(tip)

review_iter = pd.read_json(dirpath + '/review.json',chunksize = 100000,lines=True)

review_df = pd.DataFrame()

i=0
for df in review_iter:
    if i < 3:
        review_df = pd.concat([review_df, df])
        i += 1

review_data = pd.DataFrame()
label_data = pd.DataFrame()
review_data["text"] = review_df["text"].str.lower().replace('[^\w\s]','')
label_data["label"] = review_df[0:30000,"stars"]
review_list = review_data["text"].tolist()

#Tokenize
words = ' '.join(review_list[0:30000]).split()
count_words = Counter(words)
total_words = len(words)
#print("Total word count" + str(total_words))

sorted_words = count_words.most_common(total_words)

vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

reviews_int = []
for review in review_list:
    r = [vocab_to_int[w] for w in review.split()]
    reviews_int.append(r)
#print (reviews_int[0:3])

encoded_labels = np.where(label_data['label'] > 3, 1, 0)
# encoded_labels = np.array(encoded_labels)

reviews_len = [len(x) for x in reviews_int]
pd.Series(reviews_len).hist()
plt.save('review.png')
#pd.Series(reviews_len).describe()

reviews_int = [ reviews_int[i] for i, l in enumerate(reviews_len) if l>0 ]

encoded_labels = [ encoded_labels[i] for i, l in enumerate(reviews_len) if l> 0 ]


def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype = int)
    
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
    
    return features

features = pad_features(reviews_int,70)
len_feat = len(features)
split_frac = 0.8
train_x = features[0:int(split_frac*len_feat)]
train_y = encoded_labels[0:int(split_frac*len_feat)]
remaining_x = features[int(split_frac*len_feat):]
remaining_y = encoded_labels[int(split_frac*len_feat):]
valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
valid_y = remaining_y[0:int(len(remaining_y)*0.5)]
test_x = remaining_x[int(len(remaining_x)*0.5):]
test_y = remaining_y[int(len(remaining_y)*0.5):]

# create Tensor datasets
# pylint: disable=E1101
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
# pylint: enable=E1101
# dataloaders
batch_size = 50
# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2
net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train_on_gpu = True
# training params

epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        # pylint: disable=E1101
        inputs = inputs.type(torch.LongTensor)
        # pylint: enable=E1101
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()
                # pylint: disable=E1101
                inputs = inputs.type(torch.LongTensor)
                # pylint: enable=E1101
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))


# Get test data loss and accuracy

test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    
    # get predicted outputs
    # pylint: disable=E1101
    inputs = inputs.type(torch.LongTensor)
    # pylint: enable=E1101
    output, h = net(inputs, h)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    # pylint: disable=E1101
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    # pylint: enable=E1101
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))


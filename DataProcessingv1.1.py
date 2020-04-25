import json
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import re
# %matplotlib inline

dirpath = './dataset'

review = []
for line in open(dirpath + '/review.json', 'r', encoding="utf8"):
    review.append(json.loads(line))

tip = []
for line in open(dirpath + '/tip.json', 'r', encoding="utf8"):
    tip.append(json.loads(line))

review_df = pd.DataFrame.from_dict(review)
tip_df = pd.DataFrame.from_dict(tip)

review_data = review_df["text"].str.lower().replace('[^\w\s]','')

review_list = review_data["text"].tolist()

#Tokenize
all_text = ' '.join(review_list)
words = all_text.split()
count_words = Counter(words)
total_words = len(words)
print("Total word count" + str(total_words))

sorted_words = count_words.most_common(total_words)

vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

reviews_int = []
for review in review_list:
    r = [vocab_to_int[w] for w in review.split()]
    reviews_int.append(r)
print (reviews_int[0:3])

# encoded_labels = [1 if label =='positive' else 0 for label in labels_split]
# encoded_labels = np.array(encoded_labels)

reviews_len = [len(x) for x in reviews_int]
pd.Series(reviews_len).hist()
plt.savefig('./review.jpg')
pd.Series(reviews_len).describe()
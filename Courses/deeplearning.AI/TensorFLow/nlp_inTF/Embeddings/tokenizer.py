'''
Developer: vkyprmr
Filename: tokenizer.py
Created on: 2020-09-08 at 17:56:16
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-08 at 21:00:38
'''

#%%
# Imports
import json
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%%
# Tokenizing sarcasm data
with open("Data/Sarcasm/coursera/sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(f'Total words: {len(word_index)}')
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded.shape)

#%%
# Tokenizing bbc data
def read_data(filename):
    df = pd.read_csv(filename, delimiter=',')
    labels = df.category.tolist()
    sentences = df.text.tolist()

    return sentences, labels

filename = 'Data/BBC/bbc-text.csv'

# %%
sentences, labels = read_data(filename)

### Sentence tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(f'Total words: {len(word_index)}')
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded.shape)

### Label tokenizer
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index

label_seq = label_tokenizer.texts_to_sequences(labels)

print(label_seq)
print(label_word_index)

# %%

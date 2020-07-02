# How to use this script: Split vocabulary of SS3 classifier into equal chunks
# for chunk in chunks call python map.py path_to_vocab_chunks.csv path_to_chunk_closest_mapping.json
# Combine results, easiest way:
# with open('path_to_chunk_closest_mapping.json') as file:
#    mapping0 = json.load(file)
# with open('path_to_chunk_closest_mapping.json') as file:
#    mapping1 = json.load(file)
# final_mapping = {**mapping0, **mapping1}
# with open('final_mapping.json', 'w') as file:
#    json.dump(final_mapping, file)


import numpy as np
import pandas as pd
import time
import json
from scipy import spatial
from sys import argv


def find_closest_embeddings_word(embedding, num_words):
  return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))[1:num_words+1]

def read_glove(path):
   embeddings_dict = {}
   with open(path, 'r', encoding="utf-8") as files:
      for line in files:
         values = line.split()
         word = values[0]
         try:
            vector = np.asarray(values[1:], "float32")
            if len(vector) == 300: ## Unfortunately there are some entries not correct
               embeddings_dict[word] = vector
         except:
            print("Failed:", word, values[1])
   return embeddings_dict

def read_vocab(path):
   return pd.read_csv(path)

def calc_closest_map(emb_dict, vocab):
   counter = 0
   closest_mapping = {}
   for word in vocab['term']:
       print(len(vocabulary['term']) - counter, " words to go")
       counter += 1
       start = time.time()
       closest = []
       try:
          closest = find_closest_embeddings_word(embeddings_dict[word], 10)
       except:
          pass
       closest_mapping[word] = closest
       end = time.time()
       print("Time for finding word:", end-start)
   return closest_mapping

def to_file(mapping, file_name):
   with open(file_name, 'w') as file:
       json.dump(mapping, file)

print("-- Read Glove --")
embeddings_dict = read_glove("./glove.6B.300d.txt")
print("-- Read Vocab --")
vocabulary = read_vocab(argv[1])
print("-- Calculate Mapping --")
mapping = calc_closest_map(embeddings_dict, vocabulary)
print("-- Write to File --")
to_file(mapping, argv[2])
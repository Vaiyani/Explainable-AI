from nltk import word_tokenize, sent_tokenize
import os
import pandas as pd
import numpy as np
from collections import defaultdict

def avg_sent_word(samples, labels, target):
    """
    Compute Average Number of Words per Sample in Target Class
  
    Parameters:
    samples (list of strings): List containing all samples
    labels (list of strings): List contanining the respective labels per sample
    target (string): Target class, we want the average word number to know for
  
    Returns:
    Float: Average Word Number
    """
    size = 0
    for index in range(len(samples)):
        if labels[index] == target:
            size += len(word_tokenize(samples[index]))
    return size / len(samples)

def avg_sent_number(samples, labels, target):
    """
    Compute Average Number of Sentences per Sample in Target Class
  
    Parameters:
    samples (list of strings): List containing all samples
    labels (list of strings): List contanining the respective labels per sample
    target (string): Target class, we want the average word number to know for
  
    Returns:
    Float: Average Sentence Number
    """
    sentence_number = 0
    for index in range(len(samples)):
        if labels[index] == target:
            sentence_number += len(sent_tokenize(samples[index]))
    return sentence_number / len(samples)


def avg_para_number(samples, labels, target):
    """
    Compute Average Number of Paragraphs per Sample in Target Class
  
    Parameters:
    samples (list of strings): List containing all samples
    labels (list of strings): List contanining the respective labels per sample
    target (string): Target class, we want the average word number to know for
  
    Returns:
    Float: Average Paragraph Number
    """
    para_number = 0
    for index in range(len(samples)):
        if labels[index] == target:
            para_number += len(samples[index].split('\n'))
    return para_number / len(samples)

def remove_paragraph_capability(samples, labels):
    """
    Remove paragraphs from samples, remove paragraph capability from SS3 Classifier.
    
    Splits all samples containing x \in {2,..} paragraphs into x samples each containing one paragraph.
  
    Parameters:
    samples (list of strings): List containing all samples
    labels (list of strings): List contanining the respective labels per sample
  
    Returns:
    List of strings: Samples without paragraphs
    List of strings: Respective labels for the samples
    """
    par_samples = []
    par_labels = []
    for index in range(len(samples)):
        splitted_sample = samples[index].split("\n\n")
        for split in splitted_sample:
            par_samples.append(split)
            par_labels.append(labels[index])
    return par_samples, par_labels


def add_paragraph_capability(samples, labels):
    """
    Rewrites long samples, so that they contain paragraphs.
    
    If a sample contain more than two periods, it will be splitted after every second period.
    This results in paragraphs containing at max two sentences.
  
    Parameters:
    samples (list of strings): List containing all samples
    labels (list of strings): List contanining the respective labels per sample
  
    Returns:
    List of strings: Samples without paragraphs
    List of strings: Respective labels for the samples
    """
    changed = 0
    par_samples = []
    for index in range(len(samples)):
        cur_sample = samples[index]
    res_sample = ""
    while cur_sample.count('.') > 2:
        res_sample += '.'.join(cur_sample.split('.')[:2])
        res_sample += '.\n'
        cur_sample = '.'.join(cur_sample.split('.')[2:])
    if res_sample:
        changed += 1
        par_samples.append(res_sample)
    else:
        par_samples.append(cur_sample)
    return par_samples, labels


def _save_vocab(clf, path):
    """
    Saves vocabulary at given path.
    
    This is due to the fact, that PySS3 does not offer a method to access the dictonary.
  
    Parameters:
    clf (SS3 Object): Trained SS3 classifier
    path (string): Path where to store the dictonary
  
    Returns:
    None
    """
    clf.save_vocab(path=path)

def _get_vocab(vocab_dir, attribute):
    """
    Retrieve vocabulary from path.
  
    Parameters:
    vocab_dir (string): Path to vocabulary files
    attribute (string): Attribute to return for each term in voculary - either 'cv' or 'fr'
  
    Returns:
    Dictionary of Dataframes: Dictionary with classes/n-gram as key and respective Pandas Dataframe as value
    """
    list_of_vocabs = {}

    for filename in os.listdir(vocab_dir):
        csv_file = pd.read_csv(vocab_dir + filename)
        list_of_vocabs[filename] = csv_file

    for key, value in list_of_vocabs.items():
        list_of_vocabs[key] = list_of_vocabs[key][["term", attribute]]

    return list_of_vocabs


def _convert_to_df(list_of_vocabs, attribute):
    """
    Convert list_of_vocabs to pandas Dataframe containing all classes/n-grams.
  
    Parameters:
    list_of_vocabs (dictionary of dataframes): Dictionary with classes/n-gram as key and respective Pandas Dataframe as value
    attribute (string): Attribute to return for each term in voculary - either 'cv' or 'fr'
  
    Returns:
    Dataframe: Pandas Dataframe with terms as indices and classes/n-gram as column and the column value being the respective attribute
    """
    terms = []
    for key, value in list_of_vocabs.items():
        terms = list(set(list_of_vocabs[key]["term"].tolist()) | set(terms))
    pd.Series(terms).value_counts()

    new_keys = ["term"]
    for value in list(list_of_vocabs.keys()):
        if "bigram" not in value and "trigram" not in value:
            new_keys.append(value)

    data = np.zeros((len(terms), len(new_keys)))
    # Generate Empty Dataframe
    new_vocab = pd.DataFrame(data=data, columns=new_keys, dtype=float)
    new_vocab["term"] = terms
    new_vocab = new_vocab.set_index("term")

    # Fill Dataframe with values
    for key, value in list_of_vocabs.items():
        if "bigrams" in key:
            key = key.replace("bigrams", "words")
        if "trigrams" in key:
            key = key.replace("trigrams", "words")
        for index, row in value.iterrows():
            new_vocab[key][row["term"]] = row[attribute]
    return new_vocab


def _calc_word_overlap(vocab_df, important=False, a=None):
    """
    Calculate word overlap routine.
    
    This calculates the number of terms overlapping between the different classes always taking the highest number of classes.
    Example:
      ##Input
        vocab_df = 
            term  | cl_1 | cl_2 | cl_3 | cl_4
            ------|------|------|------|-----
            term1 | 0.2  | 0.3  | 0.0  | 0.0 
            term2 | 0.1  | 0.5  | 0.0  | 0.4 
            term3 | 0.4  | 0.7  | 0.0  | 0.0 
    
        important = True
        a=0.05
    
      ##Output:
        {(cl_1, cl_2): 2, (cl_1, cl_2, cl_4): 1}
  
    Parameters:
    vocab_df (dataframe): Pandas Dataframe with terms as indices and classes/n-gram as column and the column value being the respective attribute
    important (bool): Whether to compute overlap only for important words or for all words
    a (float): CV threshold, will only be used if important=True
  
    Returns:
    Dictionary: Tuples of classes are the keys, number of terms in overlap between these classes are values 
    Int: Number of terms that are in the overlap of some classes.
    """
    overlap = defaultdict(int)
    counter_all = 0
    
    for keys, row in vocab_df.iterrows():
        if not important:
            if np.count_nonzero(row) > 1:
                indices = row.to_numpy().nonzero()
                res_keys = vocab_df.keys()[indices[0]]
                overlap[tuple(res_keys)] += 1
                counter_all += 1
        else: 
            if np.count_nonzero(row.ge(a)) > 1:
                indices = np.where(row.to_numpy() > a)
                res_keys = vocab_df.keys()[indices[0]]
                overlap[tuple(res_keys)] += 1
                counter_all += 1
    return overlap, counter_all

def calc_overlap(clf, path, attribute, important=False, a=None):
    """
    Calculate word overlap based on learned vocabulary.
  
    Use the trained classifier to retrieve the vocabulary and find overlaping words between different classes.
  
    Parameters:
    clf (SS3 Object): Trained SS3 classifier
    path (str): Path to store the vocabulary data
    attribute (str): Attribute to use from vocabulary - either 'cv' or 'fr'
    important (bool): Whether to compute overlap only for important words or for all words
    a (float): CV threshold, will only be used if important=True
  
    Returns:
    Dictionary: Tuples of classes are the keys, number of terms in overlap between these classes are values 
    Int: Number of terms that are in the overlap of some classes
    Float: Ratio between all words in vocabulary and those in the overlap
    """
    _save_vocab(clf, path)
    list_of_vocabs = _get_vocab(path + '/', attribute)
    vocab_df = _convert_to_df(list_of_vocabs, attribute)
    overlap, counter = _calc_word_overlap(vocab_df, important, a)
    ratio = counter/vocab_df.shape[0]
    return overlap, counter, ratio

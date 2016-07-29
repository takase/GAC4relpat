# -*- coding: utf-8 -*-


import sys
import random
import math
import collections
import operator
import argparse
import numpy as np


MAX_SENTENCE_LENGTH = 1000 #the maximum size of a sentence (approximately maximum size)
random.seed(0)
def learn_vocab(filename):
    counter = collections.Counter()
    for line in open(filename):
        for word in line.strip().split(' '):
            counter[word] += 1
        counter['</s>'] += 1
    return counter


def read_vocab(filename):
    return collections.Counter({line.strip().split(' ')[0]: int(line.strip().split(' ')[-1]) for line in open(filename)})


def read_phrase_data(filename, word_id, reverse=False):
    phrase_ids = {}
    for line in open(filename):
        line = line.strip()
        phrase, words = line.split('\t')
        if not phrase in word_id:
            continue
        words = [word_id[w] for w in words.split(',') if w in word_id]
        if not len(words) > 1:
            #if words is not a phrase, ignore them
            continue
        if reverse:
            words.reverse()
        phrase_ids[word_id[phrase]] = words
    return phrase_ids


def read_embedding(embed_file, id_word):
    #read word embeddings represented by word2vec result format
    embed = {line.strip().split(' ')[0]: np.array(line.strip().split(' ')[1:], dtype=np.float32) for index, line in enumerate(open(embed_file)) if index > 0}
    dim = len(embed.values()[0])
    embed_matrix = np.random.uniform(-0.5 / dim, 0.5 / dim, (len(id_word), dim)).astype(np.float32)
    for index, word in enumerate(id_word):
        if word in embed:
            embed_matrix[index] = embed[word]
    return embed_matrix


def make_vocab(vocabfile='', corpus='', phrase_ids_file='', phrase_reverse=False):
    #make vocabulary, word_id, id_word, phrase_ids
    if not vocabfile and not corpus:
        sys.stderr.write('Please specify the name of vocabulary file or corpus\n')
    if not phrase_ids_file:
        sys.stderr.write('Please specify the name of file representing phrase -> words\n')
    if vocabfile:
        word_freq = read_vocab(vocabfile) #word -> freq
    else:
        word_freq = learn_vocab(corpus) #word -> freq
    word_id = {} #word -> id
    id_word = [] #id -> word
    for word, freq in sorted(word_freq.items(), key=operator.itemgetter(1, 0), reverse=True):
        word_id[word] = len(word_id)
        id_word.append(word)
    if not phrase_ids_file:
        phrase_ids = {}
    else:
        phrase_ids = read_phrase_data(phrase_ids_file, word_id, phrase_reverse) #phrase index -> word id list
    return word_freq, word_id, id_word, phrase_ids


def gen_sentence(corpus, word_id, subsample_prob=[]):
    sentence = []
    for line in open(corpus):
        words = line.strip().split(' ') + ['</s>']
        words = [word_id[w] for w in words if w in word_id]
        if subsample_prob:
            sampled_words = [w for w in words if random.random() <= subsample_prob[w]]
            words = sampled_words
        sentence.extend(words)
        if len(sentence) > MAX_SENTENCE_LENGTH:
            yield(sentence)
            sentence = []
    if len(sentence) > 0:
        yield(sentence)


def gen_each_sentence(corpus, word_id, subsample_prob=[]):
    for line in open(corpus):
        words = line.strip().split(' ') #not to use a terminal symbol
        words = [word_id[w] for w in words if w in word_id]
        if subsample_prob:
            sampled_words = [w for w in words if random.random() <= subsample_prob[w]]
            sentence = sampled_words
        if len(sentence) > 1:
            yield(sentence)


def gen_batch(corpus, word_freq, word_id, phrase_ids, batchsize=100, window=5, subsample=1e-5):
    #calculate probability for subsample
    subsample_prob = []
    if subsample > 0:
        total_freq = sum(word_freq.values())
        subsample_prob = [0] * len(word_freq) #word index -> probability of subsampling
        for word, freq in word_freq.iteritems():
            prob = float(freq) / total_freq
            p = (math.sqrt(prob / subsample) + 1) * subsample / prob
            subsample_prob[word_id[word]] = p
    instance_dict = collections.defaultdict(list) #word length -> instance list (batch)
    label_dict = collections.defaultdict(list) #word length -> surrounding word index list (batch)
    processed_num = 0 #the number of processed words
    for sentence in gen_each_sentence(corpus, word_id, subsample_prob):
    #for sentence in gen_sentence(corpus, word_id, subsample_prob):
        for position, index in enumerate(sentence):
            processed_num += 1
            win = random.randint(1, window)
            begin = max(0, position - win)
            end = min(len(sentence), position + window + 1)
            if index in phrase_ids:
                index = phrase_ids[index]
            else:
                index = [index]
            for surround_position, surround_index in enumerate(sentence[begin:end], start=begin):
                if surround_position == position:
                    continue
                instance_dict[len(index)].append(index)
                label_dict[len(index)].append(surround_index)
                if len(instance_dict[len(index)]) == batchsize:
                    yield(instance_dict[len(index)], label_dict[len(index)], processed_num)
                    instance_dict[len(index)] = []
                    label_dict[len(index)] = []


def read_phrase_test_data(filename, word_id, reverse=False):
    #read phrase test data, format: phrase1\tphrase2\tsim
    phrase_test_data = []
    phrase_max_size = 0
    for line in open(filename):
        phrase1, phrase2, sim = line.strip().split('\t')
        phrase1_words = [word_id[word] for word in phrase1.split(' ') if word in word_id]
        phrase2_words = [word_id[word] for word in phrase2.split(' ') if word in word_id]
        if reverse:
            phrase1_words.reverse()
            phrase2_words.reverse()
        phrase_max_size = max(phrase_max_size, len(phrase1_words), len(phrase2_words))
        phrase_test_data.append((phrase1_words, phrase2_words, float(sim)))
    return phrase_test_data, phrase_max_size


def read_setting(filename):
    word_freq = {}
    word_id = {}
    setting_info = {}
    for line in open(filename):
        if line.startswith('Composition_function:'):
            setting_info[line.strip().split(' ')[0][:-1]] = line.strip().split(' ')[1]
        elif line.startswith('Dim:'):
            setting_info[line.strip().split(' ')[0][:-1]] = int(line.strip().split(' ')[1])
        elif line.startswith('Phrase_reverse:'):
            setting_info[line.strip().split(' ')[0][:-1]] = bool(line.strip().split(' ')[1])
        elif line.startswith('Embed_train:'):
            setting_info[line.strip().split(' ')[0][:-1]] = bool(line.strip().split(' ')[1])
        elif line.startswith('Id:'):
            index, word, freq = line.strip().split('\t')
            index = int(index[4:])
            word = word[6:]
            freq = int(freq[6:])
            word_freq[word] = freq
            word_id[word] = index
    return word_freq, word_id, setting_info


def test(args):
    vocab_from_corpus, word_id, id_word, phrase_ids_reverse = make_vocab(corpus=args.corpus_file, phrase_ids_file=args.phrase_file, phrase_reverse=True)
    vocab_from_file, word_id, id_word, phrase_ids = make_vocab(vocabfile=args.vocab_file, phrase_ids_file=args.phrase_file)
    print len(vocab_from_file), len(vocab_from_corpus)
    for word, freq in vocab_from_corpus.most_common():
        print 'Corpus\t%s\t%s'%(word, freq)
        print 'Vocab\t%s\t%s'%(word, vocab_from_file[word])
    for i, words in enumerate(phrase_ids.iteritems()):
        if i > 5:
            break
        print words
        print phrase_ids_reverse[words[0]]
    for i, batch in enumerate(gen_batch(args.corpus_file, vocab_from_file, word_id, phrase_ids)):
        if i > 100:
            break
        instance, label = batch
        print instance
        print label
        print len(instance), len(label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phrase', dest='phrase_file', default='',
                        help='specify the name of file representing phrase and constituent words')
    parser.add_argument('-v', '--vocab', dest='vocab_file', default='',
                        help='specify the name of file representing word and frequency')
    parser.add_argument('-c', '--corpus', dest='corpus_file', default='',
                        help='specify the name of corpus file')
    args = parser.parse_args()
    test(args)


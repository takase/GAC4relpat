# -*- coding: utf-8 -*-


#learn composition function (RNN, GRU, LSTM, and GAC) by using tensorflow


import sys
import os
import operator
import time
import numpy as np
import tensorflow as tf

import utils

flags = tf.app.flags


flags.DEFINE_string('save_path', None, 'Specify the name of directory to write outputs (model and model information)')
flags.DEFINE_string('model_name', None, 'Specify the name of model name')
flags.DEFINE_string('train_data', None, 'Specify the name (path) of training corpus')
flags.DEFINE_string('phrase_data', None, 'Specify the name of file representing phrase and constituent word sequence')
flags.DEFINE_string('init_word_data', None, 'Specify the name (path) of word vectors for initialization')
flags.DEFINE_string('init_context_data', None, 'Specify the name (path) of context vectors (word vectors for prediction) for initialization')
flags.DEFINE_string('vocab', None, 'Specify the name of vocabulary file')
flags.DEFINE_bool('reverse', False, 'Reverse word sequence in a phrase or not')
flags.DEFINE_bool('save_graph', False, 'Save graph data for tensorboard illustration')
flags.DEFINE_bool('not_embedding_train', False, 'Decide to train embedding or not')
flags.DEFINE_integer('dim', 300, 'The dimension size of word embedding')
flags.DEFINE_integer('epoch_num', 1, 'Epoch number')
flags.DEFINE_float('learning_rate', 0.0025, 'Initial learning rate')
flags.DEFINE_integer('neg', 20, 'Negative samples per an instance')
flags.DEFINE_integer('batch_size', 5, 'Size of a minibatch')
flags.DEFINE_integer('window', 5, 'The window size')
flags.DEFINE_float('subsample', 1e-5, 'The subsample threshold for word occurrence')
flags.DEFINE_string('composition_function', 'RNN', 'Specify the type of composition function')
flags.DEFINE_integer('seed', 0, 'Specify the number of random seed')


FLAGS = flags.FLAGS


class Options(object):
    def __init__(self):
        self.save_path = FLAGS.save_path
        self.model_name =FLAGS.model_name
        if not self.model_name:
            self.model_name = FLAGS.composition_function + '_model'
        self.train_data = FLAGS.train_data
        self.phrase_data = FLAGS.phrase_data
        self.init_word_data = FLAGS.init_word_data
        self.init_context_data = FLAGS.init_context_data
        self.vocab = FLAGS.vocab
        self.reverse = FLAGS.reverse
        self.save_graph = FLAGS.save_graph
        self.not_embedding_train = FLAGS.not_embedding_train
        self.dim = FLAGS.dim
        self.epoch_num = FLAGS.epoch_num
        self.learning_rate = FLAGS.learning_rate
        self.neg = FLAGS.neg
        self.batch_size = FLAGS.batch_size
        self.window = FLAGS.window
        self.subsample = FLAGS.subsample
        self.composition_function = FLAGS.composition_function
        self.seed = FLAGS.seed


class LearnPhraseComposition(object):
    def __init__(self, options, session):
        self._options = options
        self._session = session
        word_freq, word_id, id_word, phrase_ids = utils.make_vocab(vocabfile=self._options.vocab, corpus=self._options.train_data, phrase_ids_file=self._options.phrase_data, phrase_reverse=self._options.reverse)
        self._word_freq = word_freq
        self._word_id = word_id
        self._id_word = id_word
        self._phrase_ids = phrase_ids
        self.save_setting()
        self.freq_table = self.make_freq_table(self._id_word, self._word_freq)
        phrase_max_size = max([len(word_seq) for word_seq in phrase_ids.values()] + [0])
        self.build_graph(phrase_max_size, self._options.composition_function, self._options.dim, self._options.batch_size,
                         self._options.neg, self._options.learning_rate, self._id_word, self.freq_table, self._options.init_word_data, 
                         self._options.init_context_data, self._options.epoch_num, not self._options.not_embedding_train)


    def save_setting(self):
        f = open(os.path.join(self._options.save_path,  self._options.model_name + '_train_setting.txt'), 'w')
        f.write('Composition_function: %s\n'%(self._options.composition_function))
        f.write('Dim: %d\n'%(self._options.dim))
        f.write('Phrase_reverse: %s\n'%(self._options.reverse))
        f.write('Embed_train: %s\n'%(not self._options.not_embedding_train))
        for index, word in enumerate(self._id_word):
            freq = self._word_freq[word]
            f.write('Id: %d\tWord: %s\tFreq: %d\n'%(index, word, freq))
        f.close()


    def make_freq_table(self, id_word, word_freq):
        return [word_freq[word] for word in id_word]


    def construct_composition(self, phrase_max_size, composition_function, dim, batch_size, embedding_train):
        holder = {}
        composed = {}
        one_word_holder = tf.placeholder(tf.int32, [batch_size, 1], '1_word_holder')
        holder[1] = one_word_holder
        one_word_holder = tf.reshape(one_word_holder, [batch_size])
        if embedding_train and composition_function != 'Add':
            one_word_embed = tf.tanh(tf.nn.embedding_lookup(self._embed, one_word_holder))
        else:
            one_word_embed = tf.nn.embedding_lookup(self._embed, one_word_holder)
        composed[1] = one_word_embed
        if composition_function == 'RNN':
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(dim) #in basicRNN, output equals state
            initial_state = rnn_cell.zero_state(batch_size, tf.float32)
        elif composition_function == 'GRU':
            rnn_cell = tf.nn.rnn_cell.GRUCell(dim)
            initial_state = rnn_cell.zero_state(batch_size, tf.float32)
        elif composition_function == 'LSTM':
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(dim, forget_bias=0.0)
            initial_state = rnn_cell.zero_state(batch_size, tf.float32)
        elif composition_function == 'GAC':
            word_iw = tf.Variable(initial_value=tf.random_uniform([dim, dim], -0.5 / dim, 0.5 / dim), name='word_iw')
            state_iw = tf.Variable(initial_value=tf.random_uniform([dim, dim], -0.5 / dim, 0.5 / dim), name='state_iw')
            bias_iw = tf.Variable(initial_value=tf.zeros([dim], tf.float32), name='bias_iw')
            word_if = tf.Variable(initial_value=tf.random_uniform([dim, dim], -0.5 / dim, 0.5 / dim), name='word_if')
            state_if = tf.Variable(initial_value=tf.random_uniform([dim, dim], -0.5 / dim, 0.5 / dim), name='state_if')
            bias_if = tf.Variable(initial_value=tf.zeros([dim], tf.float32), name='bias_if')
            initial_state = tf.zeros([batch_size, dim], tf.float32)
        elif composition_function == 'CNN':
            weight_conv = tf.Variable(initial_value=tf.random_uniform([3, dim, 1, dim], -0.5 / dim, 0.5 / dim), name='weight_conv')
            bias_conv = tf.Variable(initial_value=tf.zeros([dim], tf.float32), name='bias_conv')
        for i in xrange(2, phrase_max_size+1):
            phrase_holder = tf.placeholder(tf.int32, [batch_size, i], '%s_word_holder'%i)
            holder[i] = phrase_holder
            embed = tf.nn.embedding_lookup(self._embed, phrase_holder)
            if composition_function in ['RNN', 'GRU', 'LSTM']:
                state = initial_state
                with tf.variable_scope('RNN') as scope:
                    tf.get_variable_scope().set_initializer(tf.random_uniform_initializer(minval=-0.5 / dim, maxval=0.5 / dim)) #initialize weight matrix of RNN by this initializer
                    for step in xrange(i):
                        if step > 0 or i > 2: tf.get_variable_scope().reuse_variables()#to reuse variable in RNN
                        output, state = rnn_cell(embed[:, step, :], state)
            elif composition_function == 'GAC':
                state = initial_state
                for step in xrange(i):
                    input_embed = embed[:, step, :]
                    input_gate = tf.sigmoid(tf.matmul(input_embed, word_iw) + tf.matmul(state, state_iw) + bias_iw)
                    forget_gate = tf.sigmoid(tf.matmul(input_embed, word_if) + tf.matmul(state, state_if) + bias_if)
                    state = tf.tanh(tf.mul(input_gate, input_embed) + tf.mul(forget_gate, state))
                output = state
            elif composition_function == 'CNN':
                embed = tf.pad(tf.reshape(embed, [batch_size, i, dim, 1]), [[0, 0], [1, 1], [0, 0], [0, 0]], mode='CONSTANT') #padding by zero vector
                conv = tf.tanh(tf.nn.conv2d(embed, weight_conv, [1, 1, dim, 1], 'SAME') + bias_conv)
                max_pooled = tf.nn.max_pool(conv, [1, i+2, 1, 1], [1, i+2, 1, 1], 'SAME')
                output = tf.reshape(max_pooled, [batch_size, dim])
            elif composition_function == 'Add':
                output = tf.reduce_mean(embed, 1)
            composed[i] = output
        return holder, composed


    def forward(self, phrase_max_size=1, composition_function='RNN', dim=100, batch_size=1, neg=1,
                freq_table=[], init_word_matrix=[], init_context_matrix=[], embedding_train=False):
        if len(init_word_matrix) > 0:
            embed = tf.Variable(initial_value=init_word_matrix, trainable=embedding_train, name='embed')
        else:
            embed = tf.Variable(initial_value=tf.random_uniform([len(freq_table), dim], -0.5 / dim, 0.5 / dim),
                                trainable=embedding_train, name='embed')
        self._embed = embed
        if len(init_context_matrix) > 0:
            context_emb = tf.Variable(initial_value=init_context_matrix, trainable=embedding_train, name='context_emb')
        else:
            context_emb = tf.Variable(initial_value=tf.zeros([len(freq_table), dim]),
                                trainable=embedding_train, name='context_emb')
        self._context_emb = context_emb
        holder, composed = self.construct_composition(phrase_max_size, composition_function, dim, batch_size, embedding_train)
        context_id = tf.placeholder(tf.int32, [batch_size])
        #negative sampling for each example
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.reshape(tf.cast(context_id, dtype=tf.int64), [batch_size, 1]),
            num_true=1,
            num_sampled=neg+1,
            unique=True,
            range_max=len(freq_table),
            distortion=0.75,
            unigrams=freq_table))
        exclude_list = tf.cast(tf.constant([self._word_id['</s>']]), dtype=tf.int64) #exclude terminal node from negative sampling result
        sampled_ids, _ = tf.listdiff(sampled_ids, exclude_list)
        true_logit = {}
        negative_logit = {}
        true_context = tf.nn.embedding_lookup(self._context_emb, context_id)
        negative_context = tf.nn.embedding_lookup(self._context_emb, sampled_ids[:neg])
        for length, compose_embed in composed.iteritems():
            #calculate prob for positive example
            true_logit[length] = tf.reduce_sum(tf.mul(compose_embed, true_context), 1)
            negative_logit[length] = tf.matmul(compose_embed, negative_context, transpose_b=True)
        return holder, composed, context_id, true_logit, negative_logit


    def nce_loss(self, true_logit, negative_logit, batch_size, embedding_train):
        #build loss ops
        loss = {}
        for length in true_logit:
            if length == 1 and not embedding_train:
                #skip because this situation is not necessary to calculate loss
                continue
            true_xent = tf.nn.sigmoid_cross_entropy_with_logits(true_logit[length], tf.ones_like(true_logit[length]))
            negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(negative_logit[length], tf.zeros_like(negative_logit[length]))
            loss[length] = (tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)) / batch_size
        return loss


    def optimizer(self, loss, all_word_num, epoch_num, initial_learning_rate):
        #build optimizer
        processed_num = tf.placeholder(tf.float32)
        train_word_num = all_word_num * epoch_num
        lr = initial_learning_rate * tf.maximum(0.0001, 1.0 - processed_num / train_word_num)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        optimize_op = {}
        for length in loss:
            optimize_op[length] = optimizer.minimize(loss[length])
        return processed_num, lr, optimize_op


    def build_graph(self, phrase_max_size=1, composition_function='RNN', dim=100, batch_size=1, neg=1, learning_rate=0.2, id_word=[],
                    freq_table=[], init_word_data=[], init_context_data=[], epoch_num=1, embedding_train=False):
        if init_word_data:
            init_word_matrix = utils.read_embedding(init_word_data, id_word)
        else:
            init_word_matrix = []
        if init_context_data:
            init_context_matrix = utils.read_embedding(init_context_data, id_word)
        else:
            init_context_matrix = []
        holder, composed, context_id, true_logit, negative_logit = self.forward(phrase_max_size, composition_function, dim, batch_size, neg, 
                                                                                freq_table, init_word_matrix, init_context_matrix, embedding_train)
        self.holder = holder
        self.composed = composed
        self.context_id = context_id
        self.true_logit = true_logit
        self.negative_logit = negative_logit
        loss = self.nce_loss(true_logit, negative_logit, batch_size, embedding_train)
        self.loss = loss
        for length in loss:
            tf.scalar_summary('%s_SGNS_loss'%length, loss[length])
        processed_num, lr, optimize_op = self.optimizer(loss, sum(freq_table), epoch_num, initial_learning_rate=learning_rate)
        self.processed_num = processed_num
        self.optimize_op = optimize_op
        self.lr = lr
        #initialize all values
        tf.initialize_all_variables().run()


    def train(self, current_epoch):
        #process one epoch
        sum_loss = 0.0
        current_epoch = np.array(current_epoch, np.float32)
        processed_batch = 0
        processed_before_batch = 0
        begin = time.time()
        log_begin = begin
        for instance_info in utils.gen_batch(self._options.train_data, self._word_freq, self._word_id, self._phrase_ids, 
                                             self._options.batch_size, self._options.window, self._options.subsample):
            embed_id, context_id, processed_num = instance_info
            phrase_length = len(embed_id[0])
            if self._options.not_embedding_train and phrase_length == 1:
                #if not embedding train is true, ignore one word
                continue
            loss, _, lr = self._session.run([self.loss[phrase_length], self.optimize_op[phrase_length], self.lr], 
                                            {self.holder[phrase_length]: np.array(embed_id, dtype=np.int32),
                                             self.context_id: np.array(context_id, dtype=np.int32),
                                             self.processed_num: np.array(processed_num, dtype=np.float32),
                                         })
            sum_loss += loss * self._options.batch_size
            processed_batch += 1
            #output log
            if processed_batch % 10000 == 0:
                end = time.time()
                print 'Epoch: %d\tTrained: %s\tLr: %.6f\tLoss: %.4f\tword/sec: %d'%(current_epoch, processed_num, lr, sum_loss / (processed_num - processed_before_batch), (processed_num - processed_before_batch) / (end - log_begin))
                log_begin = time.time()
                processed_before_batch = processed_num
                sum_loss = 0.0


def main(_):
    if not FLAGS.train_data or not FLAGS.save_path:
        sys.stderr.write('Please specify training file (--train_data) and save path (--save_path)\n')
        sys.exit(1)
    if FLAGS.not_embedding_train and (not FLAGS.init_word_data or not FLAGS.init_context_data):
        sys.stderr.write('If you do not train embedding, you must specify initialize vector files (--init_word_data and --init_context_data)\n')
        sys.exit(1)
    if not FLAGS.phrase_data:
        sys.stderr.write('Please specify phrase file (--phrase_data)\n')
        sys.exit(1)
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        tf.set_random_seed(opts.seed)
        with tf.device('/cpu:0'):
            #construct model class (building graph)
            model = LearnPhraseComposition(opts, session)
        #launch saver
        saver = tf.train.Saver()
        if opts.save_graph:
            #launch summary writer
            summary_writer = tf.train.SummaryWriter(os.path.join(opts.save_path, 'data'), graph_def=session.graph_def)
            summary_writer.add_graph(session.graph_def)
        for current_epoch in xrange(1, FLAGS.epoch_num+1):
            model.train(current_epoch)
        saver.save(session, os.path.join(opts.save_path, opts.model_name + '.ckpt'))


if __name__ == '__main__':
    tf.app.run()

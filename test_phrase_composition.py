# -*- coding: utf-8 -*-


import numpy as np
import scipy.stats as stats
import tensorflow as tf

from learn_phrase_composition import LearnPhraseComposition
import utils


flags = tf.app.flags


flags.DEFINE_string('test_data', None, 'Specify the name of test data')
flags.DEFINE_string('setting', None, 'Specify the name of file representing setting data')

FLAGS = flags.FLAGS


class Options(object):
    def __init__(self):
        self.test_data = FLAGS.test_data
        self.setting = FLAGS.setting
        self.model_name = FLAGS.model_name


class TestPhraseComposition(LearnPhraseComposition):
    def __init__(self, word_freq, word_id, setting_info, phrase_max_size, session):
        self._session = session
        id_word = [''] * len(word_id)
        self._word_id = {}
        for word, index in word_id.iteritems():
            id_word[index] = word
            self._word_id[word] = index
        self._id_word = id_word
        self.freq_table = self.make_freq_table(id_word, word_freq)
        self.build_graph(phrase_max_size, setting_info['Composition_function'], setting_info['Dim'], 1, freq_table=self.freq_table, embedding_train=setting_info['Embed_train'])


def main(_):
    if not FLAGS.test_data or not FLAGS.setting or not FLAGS.model_name:
        sys.stderr.write('Please specify test data (--test_data), setting data (--setting), and model file name (--model_name)\n')
        sys.exit(1)
    opts = Options()
    word_freq, word_id, setting_info = utils.read_setting(opts.setting)
    phrase_test_data, phrase_max_size = utils.read_phrase_test_data(opts.test_data, word_id, setting_info['Phrase_reverse'])
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device('/cpu:0'):
            #construct model class (building graph)
            model = TestPhraseComposition(word_freq, word_id, setting_info, phrase_max_size, session)
        saver = tf.train.Saver()
        saver.restore(session, opts.model_name)
        acc_data = []
        sys_data = []
        for phrase_data in phrase_test_data:
            phrase1_words, phrase2_words, sim = phrase_data
            acc_data.append(sim)
            if not phrase1_words or not phrase2_words:
                sys_data.append(0.0)
                continue
            phrase1_vec = session.run([model.composed[len(phrase1_words)]],
                                      {model.holder[len(phrase1_words)]: np.array([phrase1_words], dtype=np.int32)})
            phrase1_vec = phrase1_vec[0].flatten()
            phrase2_vec = session.run([model.composed[len(phrase2_words)]],
                                      {model.holder[len(phrase2_words)]: np.array([phrase2_words], dtype=np.int32)})
            phrase2_vec = phrase2_vec[0].flatten()
            sim = np.dot(phrase1_vec / np.linalg.norm(phrase1_vec), phrase2_vec / np.linalg.norm(phrase2_vec))
            sys_data.append(sim)
    spearman = stats.spearmanr(acc_data, sys_data)
    print spearman[0]


if __name__ == '__main__':
    tf.app.run()







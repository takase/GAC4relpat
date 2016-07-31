# GAC4relpat

This project contains a program which can train Gated Additive Composition from the paper.

     Composing Distributed Representations of Relational Patterns.
     Sho Takase, Naoaki Okazaki, Kentaro Inui.

## Requirement

* Python 2.7
* Numpy
* Scipy
* Tensorflow

In addition, you have to make a vocabulary file and a phrase data file. The vocabulary file represents words and their frequencies in training corpus. The format of the vocabulary file is the same as one made by word2vec. Please see example.vocab, it is an example. The phrase data file represents phrases and their constituent words. Please see example.phrase.  
If you need a vocabulary file, phrase data file, and a training corpus used in Takase et al., 2016, please let me know (takase at ecei.tohoku.ac.jp).

Moreover, if you use not_embedding_train option, you have to prepare the results of word2vec (syn0 and syn1neg).

## Training

```
python learn_phrase_composition_tensorflow.py --train_data=training_corpus  --save_path=save_directory_name --vocab=vocabulary_file_name
--phrase_data=phrase_data_name --dim=300 --epoch_num=1 --learning_rate=0.0025 --neg=20 --batch_size=5 --window=5
--subsample=1e-5 --composition_function=GAC --model_name=model_name_for_saving
--init_word_data=word_vector(syn0)_for_initialization --init_context_data=word_vector(syn1neg)_for_initialization --not_embedding_train
```

## Test

test_phrase_composition.py calculates the Spearman's rank correlation between the similarities provided by a test data and the cosine similarities calculated by encoders. Before running test_phrase_composition.py, please prepare a test data. In the test data, constituent words of a phrase are joined to each other by '_'. The relational pattern similarity dataset is [here](https://github.com/takase/relPatSim).

```
python test_phrase_composition.py --model_name=trained_model_file_name  --test_data=test_data --setting=training_setting_file_name
```


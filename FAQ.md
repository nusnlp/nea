# Frequently Asked Questions #

1. How to run the training code?

You can use the following command to train a model:

```bash
THEANO_FLAGS="device=gpu0,floatX=float32" python train_nea.py
	-tr data/fold_0/train.tsv
	-tu data/fold_0/dev.tsv
	-ts data/fold_0/test.tsv
	-p 1	# Prompt ID
	-o output_dir
```

2. How to see the available options for running ```train_nea.py``` script?

```
python train_nea.py -h
```

3. What is ```--emb```?

You can use ```--emb``` option to initialize the lookup table layer with pre-trained embeddings:

```bash
THEANO_FLAGS="device=gpu0,floatX=float32" python train_nea.py
	-tr data/fold_0/train.tsv
	-tu data/fold_0/dev.tsv
	-ts data/fold_0/test.tsv
	-p 1	# Prompt ID
	--emb embeddings.w2v.txt
	-o output_dir
```

4. What is the file format of ```embeddings.w2v.txt```?

The format of this file is the simple Word2Vec format. The first line should include the number of rows and columns of the word embeddings matrix.

5. Which pre-trained word embeddings should I use?

```--emb``` is optinal. If you want to replicate our results, download [this file](http://ai.stanford.edu/~wzou/mt/biling_mt_release.tar.gz). Convert ```En_vectors.txt``` to Word2Vec format and use if with ```--emb``` option. To convert it to Word2Vec format, simply add the W2V header to the file, like this:

```
100229 50
the -0.45485 1.0028 -1.4068 ...
, -0.4088 -0.10933 -0.099279 ...
. -0.58359 0.41348 -0.70819 ...
...
```


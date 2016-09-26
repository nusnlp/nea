# Neural Essay Assessor #

An automatic essay scoring system based on convolutional and recurrent neural networks, including GRU and LSTM.

### Set Up ###

* Install Keras
* Run train_nea.py

### Options ###

You can see the list of available options by running:
```bash
python train_nea.py -h
```
### Example ###

```bash
THEANO_FLAGS="device=gpu0,floatX=float32" ~/git/nea-released/train_nea.py
	-tr train.tsv
	-tu dev.tsv
	-ts test.tsv
	-p 1	# Prompt ID
	--emb embeddings.w2v.txt
	-o output_dir
```


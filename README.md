# Neural Essay Assessor 1.0 #

An automatic essay scoring system based on convolutional and recurrent neural networks, including GRU and LSTM.

### Set Up ###

* Install Keras (with Theano backend)
* Prepare data
* Run train_nea.py

### Environment

- Keras 1.1.0
- Theano 0.8.2
- Python 2.7.9
- Numpy 1.13.3
- Scipy 0.19.1

### Data ###

We have used 5-fold cross validation on ASAP dataset to evaluate our system. This dataset (training_set_rel3.tsv) can be downloaded from [here](https://www.kaggle.com/c/asap-aes/data). After downloading the file, put it in the [data](https://github.com/nusnlp/nea/tree/master/data) directory and create training, development and test data using ```preprocess_asap.py``` script:

```bash
cd data
python preprocess_asap.py -i training_set_rel3.tsv
```

### Options ###

You can see the list of available options by running:
```bash
python train_nea.py -h
```
### Example ###

The following command trains a model for prompt 1 in the ASAP dataset, using the training and development data from fold 0 and evaluates it.

```bash
THEANO_FLAGS="device=gpu0,floatX=float32" python train_nea.py
	-tr data/fold_0/train.tsv
	-tu data/fold_0/dev.tsv
	-ts data/fold_0/test.tsv
	-p 1	# Prompt ID
	--emb embeddings.w2v.txt
	-o output_dir
```

### Frequently Asked Questions ###

See our [FAQ](https://github.com/nusnlp/nea/blob/master/FAQ.md) page for a list of frequently asked questions. If the answer to your question is not there, contact me (kaveh@comp.nus.edu.sg).

### License ###

Neural Essay Assessor is licensed under the GNU General Public License Version 3. Separate commercial licensing is also available. For more information contact:

* Kaveh Taghipour (kaveh@comp.nus.edu.sg)
* Hwee Tou Ng (nght@comp.nus.edu.sg)
* Shengjia Yan (i@yanshengjia.com)

### Publication ###

Kaveh Taghipour and Hwee Tou Ng. 2016. [A neural approach to automated essay scoring](http://aclweb.org/anthology/D/D16/D16-1193.pdf). In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing.

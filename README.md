# Neural Essay Assessor #

An automatic essay scoring system based on convolutional and recurrent neural networks, including GRU and LSTM.

### Set Up ###

* Install Keras
* Prepare data
* Run train_nea.py

### Data ###

We have used 5-fold cross validation on ASAP dataset to evaluate our system. This dataset (training_set_rel3.tsv) can be downloaded from [here](https://www.kaggle.com/c/asap-aes/data). After downloading the dataset, create training, development and test data according to the essay IDs in the [data directory](https://github.com/nusnlp/nea/tree/master/data). You should keep the TSV header in all the generated files.

### Options ###

You can see the list of available options by running:
```bash
python train_nea.py -h
```
### Example ###

The following command trains a model for prompt 1 in the ASAP dataset, using the training and development data from fold 0 and evaluates it.

```bash
THEANO_FLAGS="device=gpu0,floatX=float32" ~/git/nea-released/train_nea.py
	-tr fold_0/train.tsv
	-tu fold_0/dev.tsv
	-ts fold_0/test.tsv
	-p 1	# Prompt ID
	--emb embeddings.w2v.txt
	-o output_dir
```

### License ###

Neural Essay Assessor is licensed under the GNU General Public License Version 3. Separate commercial licensing is also available. For more information contact:

* Kaveh Taghipour (kaveh@comp.nus.edu.sg)
* Hwee Tou Ng (nght@comp.nus.edu.sg)

### Publication ###

Kaveh Taghipour and Hwee Tou Ng. 2016. [A neural approach to automated essay scoring](http://www.comp.nus.edu.sg/~kaveh/papers/aesnn-emnlp16.pdf). In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing.

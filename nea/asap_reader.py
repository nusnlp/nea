import random
import codecs
import sys
import nltk
import logging
import re
import numpy as np
import pickle as pk

logger = logging.getLogger(__name__)
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

asap_ranges = {
	0: (0, 60),
	1: (2,12),
	2: (1,6),
	3: (0,3),
	4: (0,3),
	5: (0,4),
	6: (0,4),
	7: (0,30),
	8: (0,60)
}

def get_ref_dtype():
	return ref_scores_dtype

def tokenize(string):
	tokens = nltk.word_tokenize(string)
	for index, token in enumerate(tokens):
		if token == '@' and (index+1) < len(tokens):
			tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
			tokens.pop(index)
	return tokens

def get_score_range(prompt_id):
	return asap_ranges[prompt_id]

def get_model_friendly_scores(scores_array, prompt_id_array):
	arg_type = type(prompt_id_array)
	assert arg_type in {int, np.ndarray}
	if arg_type is int:
		low, high = asap_ranges[prompt_id_array]
		scores_array = (scores_array - low) / (high - low)
	else:
		assert scores_array.shape[0] == prompt_id_array.shape[0]
		dim = scores_array.shape[0]
		low = np.zeros(dim)
		high = np.zeros(dim)
		for ii in range(dim):
			low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
		scores_array = (scores_array - low) / (high - low)
	assert np.all(scores_array >= 0) and np.all(scores_array <= 1)
	return scores_array

def convert_to_dataset_friendly_scores(scores_array, prompt_id_array):
	arg_type = type(prompt_id_array)
	assert arg_type in {int, np.ndarray}
	if arg_type is int:
		low, high = asap_ranges[prompt_id_array]
		scores_array = scores_array * (high - low) + low
		assert np.all(scores_array >= low) and np.all(scores_array <= high)
	else:
		assert scores_array.shape[0] == prompt_id_array.shape[0]
		dim = scores_array.shape[0]
		low = np.zeros(dim)
		high = np.zeros(dim)
		for ii in range(dim):
			low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
		scores_array = scores_array * (high - low) + low
	return scores_array

def is_number(token):
	return bool(num_regex.match(token))

def load_vocab(vocab_path):
	logger.info('Loading vocabulary from: ' + vocab_path)
	with open(vocab_path, 'rb') as vocab_file:
		vocab = pk.load(vocab_file)
	return vocab

def create_vocab(file_path, prompt_id, maxlen, vocab_size, tokenize_text, to_lower):
	logger.info('Creating vocabulary from: ' + file_path)
	if maxlen > 0:
		logger.info('  Removing sequences with more than ' + str(maxlen) + ' words')
	total_words, unique_words = 0, 0
	word_freqs = {}
	with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
		input_file.next()
		for line in input_file:
			tokens = line.strip().split('\t')
			essay_id = int(tokens[0])
			essay_set = int(tokens[1])
			content = tokens[2].strip()
			score = float(tokens[6])
			if essay_set == prompt_id or prompt_id <= 0:
				if to_lower:
					content = content.lower()
				if tokenize_text:
					content = tokenize(content)
				else:
					content = content.split()
				if maxlen > 0 and len(content) > maxlen:
					continue
				for word in content:
					try:
						word_freqs[word] += 1
					except KeyError:
						unique_words += 1
						word_freqs[word] = 1
					total_words += 1
	logger.info('  %i total words, %i unique words' % (total_words, unique_words))
	import operator
	sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
	if vocab_size <= 0:
		# Choose vocab size automatically by removing all singletons
		vocab_size = 0
		for word, freq in sorted_word_freqs:
			if freq > 1:
				vocab_size += 1
	vocab = {'<pad>':0, '<unk>':1, '<num>':2}
	vcb_len = len(vocab)
	index = vcb_len
	for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
		vocab[word] = index
		index += 1
	return vocab

def read_essays(file_path, prompt_id):
	logger.info('Reading tsv from: ' + file_path)
	essays_list = []
	essays_ids = []
	with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
		input_file.next()
		for line in input_file:
			tokens = line.strip().split('\t')
			if int(tokens[1]) == prompt_id or prompt_id <= 0:
				essays_list.append(tokens[2].strip())
				essays_ids.append(int(tokens[0]))
	return essays_list, essays_ids

def read_dataset(file_path, prompt_id, maxlen, vocab, tokenize_text, to_lower, score_index=6, char_level=False):
	logger.info('Reading dataset from: ' + file_path)
	if maxlen > 0:
		logger.info('  Removing sequences with more than ' + str(maxlen) + ' words')
	data_x, data_y, prompt_ids = [], [], []
	num_hit, unk_hit, total = 0., 0., 0.
	maxlen_x = -1
	with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
		input_file.next()
		for line in input_file:
			tokens = line.strip().split('\t')
			essay_id = int(tokens[0])
			essay_set = int(tokens[1])
			content = tokens[2].strip()
			score = float(tokens[score_index])
			if essay_set == prompt_id or prompt_id <= 0:
				if to_lower:
					content = content.lower()
				if char_level:
					#content = list(content)
					raise NotImplementedError
				else:
					if tokenize_text:
						content = tokenize(content)
					else:
						content = content.split()
				if maxlen > 0 and len(content) > maxlen:
					continue
				indices = []
				if char_level:
					raise NotImplementedError
				else:
					for word in content:
						if is_number(word):
							indices.append(vocab['<num>'])
							num_hit += 1
						elif word in vocab:
							indices.append(vocab[word])
						else:
							indices.append(vocab['<unk>'])
							unk_hit += 1
						total += 1
				data_x.append(indices)
				data_y.append(score)
				prompt_ids.append(essay_set)
				if maxlen_x < len(indices):
					maxlen_x = len(indices)
	logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
	return data_x, data_y, prompt_ids, maxlen_x

def get_data(paths, prompt_id, vocab_size, maxlen, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None, score_index=6):
	train_path, dev_path, test_path = paths[0], paths[1], paths[2]
	
	if not vocab_path:
		vocab = create_vocab(train_path, prompt_id, maxlen, vocab_size, tokenize_text, to_lower)
		if len(vocab) < vocab_size:
			logger.warning('The vocabualry includes only %i words (less than %i)' % (len(vocab), vocab_size))
		else:
			assert vocab_size == 0 or len(vocab) == vocab_size
	else:
		vocab = load_vocab(vocab_path)
		if len(vocab) != vocab_size:
			logger.warning('The vocabualry includes %i words which is different from given: %i' % (len(vocab), vocab_size))
	logger.info('  Vocab size: %i' % (len(vocab)))
	
	train_x, train_y, train_prompts, train_maxlen = read_dataset(train_path, prompt_id, maxlen, vocab, tokenize_text, to_lower)
	dev_x, dev_y, dev_prompts, dev_maxlen = read_dataset(dev_path, prompt_id, 0, vocab, tokenize_text, to_lower)
	test_x, test_y, test_prompts, test_maxlen = read_dataset(test_path, prompt_id, 0, vocab, tokenize_text, to_lower)
	
	overal_maxlen = max(train_maxlen, dev_maxlen, test_maxlen)
	
	return ((train_x,train_y,train_prompts), (dev_x,dev_y,dev_prompts), (test_x,test_y,test_prompts), vocab, len(vocab), overal_maxlen, 1)

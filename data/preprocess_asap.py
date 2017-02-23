#!/usr/bin/python

## Script to pre-process ASAP dataset (training_set_rel3.tsv) based on the essay IDs

import argparse
import codecs
import sys

def extract_based_on_ids(dataset, id_file):
	lines = []
	with open(id_file) as f:
		for line in f:
			line = line.strip()
			try:
				lines.append(dataset[line])
			except:
				print >> sys.stederr, "ERROR:Invalid ID " + line + "in " +id_file
	return lines

def create_dataset(lines, output_fname):
	f_write = open(output_fname,'w')
	f_write.write(dataset['header'])
	for line in lines:
		f_write.write(line.decode('cp1252','replace').encode('utf-8'))
		
def collect_dataset(input_file):
	dataset = dict()
	lcount = 0
	with open(input_file) as f:
		for line in f:
			lcount += 1
			if lcount==1:
				dataset['header'] = line
				continue
			parts = line.split('\t')
			assert len(parts)>=6, "ERROR" + line
			dataset[parts[0]] = line
	return dataset

parser = argparse.ArgumentParser()
parser.add_argument('-i',"--input-file", dest="input_file", required=True, help="Input TSV file")
args = parser.parse_args()

dataset = collect_dataset(args.input_file)
for fold_idx in xrange(0,5):
	for dataset_type in ['dev','test','train']:
		lines = extract_based_on_ids(dataset, 'fold_'+str(fold_idx)+'/'+dataset_type+'_ids.txt')
		create_dataset(lines, 'fold_'+str(fold_idx)+'/'+dataset_type+'.tsv')


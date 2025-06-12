import argparse
import copy
import os
import glob
import numpy as np
import scipy
import pickle
import tensorflow as tf
from tensorflow import keras
import statsmodels.api as sm

def make_dataset(name, seqfile, settings):
    if not settings.initial_dataset_type in ['file', 'random']:
        raise RuntimeError('Unrecognized initial dataset type: ' + settings.initial_dataset_type)

    def encode(examples):
        # Encode sequence with mutations as a list of integers
        ones = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
        return [[ones.index(res) for res in seq] for seq in examples]

    data_lines = open(settings.initial_dataset_file, 'r').readlines()

    examples_raw = [line.split()[0] for line in data_lines]
    try:
        assert len(examples_raw) == len(set(examples_raw))
    except AssertionError:
        raise RuntimeError('Examples in file ' + settings.initial_dataset_file + ' are not all unique')

    examples = encode(examples_raw)

    labels = [[float(item) for item in line.replace(',', '').replace('[', '').replace(']', '').split()[1:]] for line in data_lines]
    assert len(examples) == len(labels)

    if seqfile and settings.campaign == 'from_file':
        seqs_lines = open(seqfile, 'r').readlines()
        seqs = [line.split()[0] for line in seqs_lines]
        new_examples = []
        new_labels = []
        new_examples_raw = []
        for seq in seqs:
            ii = examples_raw.index(seq)
            new_examples.append(examples[ii])
            new_labels.append(labels[ii])
            new_examples_raw.append(examples_raw[ii])

        examples = np.array(new_examples)
        labels = np.array(new_labels)
        examples_raw = new_examples_raw
    else:
        if settings.rng_seed < 0:
            rng_seed = None
        else:
            rng_seed = settings.rng_seed
        rng = np.random.default_rng(rng_seed)

        indices = [ind for ind in range(len(examples))]
        rng.shuffle(indices)
        examples = [examples[ind] for ind in indices]
        labels = [labels[ind] for ind in indices]
        examples_raw = [examples_raw[ind] for ind in indices]

        examples = np.array(examples[:settings.step_size])
        labels = np.array(labels[:settings.step_size])
        examples_raw = examples_raw[:settings.step_size]

    dataset = {'examples': examples, 'labels': labels}

    if settings.normalize_training_data:
        # labels = np.array([(label - np.mean(labels)) / np.sqrt(np.var(labels)) for label in labels])
        norm_labels = [np.exp(label) for label in labels]
        # norm_labels = np.array([(label - np.mean(labels)) / np.sqrt(np.var(labels)) for label in norm_labels])
        norm_labels = np.array([(label - min(labels)) / (max(labels) - min(labels)) for label in norm_labels])

        dataset = {'examples': examples, 'labels': norm_labels, 'raw_labels': labels}   # to support reporting raw labels in output file

    pickle.dump(dataset, open(name + '.pkl', 'wb'))
    return name + '.pkl', examples_raw

def update_output(dataset_file, step_number, settings):
    # Write dataset labels to new line in output file
    if not os.path.exists(settings.output_filename):
        open(settings.output_filename, 'w').write('Round\tLabels\n')

    dataset = pickle.load(open(dataset_file, 'rb'))
    if 'raw_labels' in dataset.keys():  # happens if training data were normalized
        labels = ' '.join([str(label[0]) for label in dataset['raw_labels']])
    else:
        labels = ' '.join([str(label[0]) for label in dataset['labels']])
    open(settings.output_filename, 'a').write(str(step_number) + '\t' + labels + '\n')

def get_models(settings):
    # Load pre-trained models specified in settings.biophysics_models
    # todo: if publishing this I'll probably want to supporting training these models in this program, too.
    # todo: that probably needs to include some automated process of hyperparameter optimization?
    models = glob.glob(settings.path_to_pretrained_models + '*.keras')
    models.sort()

    if not models:
        print('WARNING: no .keras files found in ' + settings.path_to_pretrained_models + '. Continuing with no '
              'pretrained models.')

    if settings.prosst > 0:
        if not os.path.exists(settings.path_to_prosst_model):
            if settings.prosst not in [20, 128, 512, 1024, 2048, 4096]:
                raise RuntimeError('prosst > 0, but was not a valid vocabulary size. Must be one of 20, 128, 512, 1024, 2048, 4096')
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            from prosst.structure.quantizer import PdbQuantizer
            from Bio import SeqIO
            import torch
            import pandas as pd

            deprot = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-" + str(settings.prosst), trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-" + str(settings.prosst), trust_remote_code=True)
            processor = PdbQuantizer()
            residue_sequence = str(SeqIO.read(settings.prosst_fasta_file, 'fasta').seq)
            structure_sequence = processor(settings.prosst_pdb_file)
            structure_sequence_offset = [i + 3 for i in structure_sequence]
            tokenized_res = tokenizer([residue_sequence], return_tensors='pt')
            input_ids = tokenized_res['input_ids']
            attention_mask = tokenized_res['attention_mask']
            structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                outputs = deprot(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ss_input_ids=structure_input_ids
                )
            logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()
            vocab = tokenizer.get_vocab()

            pickle.dump({'logits': logits, 'vocab': vocab}, open('prosst.pkl','wb'))

            models.append('prosst.pkl')
        else:
            models.append(settings.path_to_prosst_model)

    return models

def generate_correlations(dataset, models, step, settings):
    dataset_dict = pickle.load(open(dataset, 'rb'))
    dataset_labels = dataset_dict['labels']
    train = tf.data.Dataset.from_tensor_slices((dataset_dict['examples'], dataset_dict['labels'][:, 0]))
    train = train.batch(settings.batch_size)

    # Slice separately to support train_split setting
    test = tf.data.Dataset.from_tensor_slices((dataset_dict['examples'][int(len(dataset_dict['labels']) * settings.train_split):], dataset_dict['labels'][int(len(dataset_dict['labels']) * settings.train_split):, 0]))
    test = test.batch(settings.batch_size)
    test_labels = dataset_dict['labels'][int(len(dataset_dict['labels']) * settings.train_split):, 0]

    correlations = {}  # initialize list of correlations of labels with each biophysics prediction
    labels = []

    def decode(encoded):
        # Decode sequence with mutations as a list of integers

        ones = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'] 

        return ''.join([ones[ii] for ii in encoded])

    for model in models:
        if '/' in model:
            label = model[model.rindex('/') + 1:]
        else:
            label = model
        labels.append(label)

        assert not label in list(correlations.keys())

        ### Load custom layer type here, unsure if this is actually needed ###
        class ConstantLayer(tf.keras.layers.Layer):
            def __init__(self, units=32):
                super().__init__()
                self.units = units

            # Create the state of the layer (weights)
            def build(self, input_shape):
                self.kernel = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer="glorot_uniform",
                    trainable=True,
                    name="kernel",
                )
                self.bias = self.add_weight(
                    shape=(self.units,),
                    initializer="zeros",
                    trainable=True,
                    name="bias",
                )

            # Defines the computation
            def call(self, inputs):
                output_shape = tf.shape(inputs)[0]
                return tf.zeros((output_shape, 1), dtype=inputs.dtype)

        def linregress(t):
            # Helper function
            x = t[0]
            y = t[1]
            if len(x) == 0:
                to_return = argparse.Namespace()
                to_return.rvalue = 0
                to_return.intercept = 0
                to_return.slope = 0
                to_return.stderr = 0
                return to_return
            if not settings.weighted_regression:
                return scipy.stats.linregress(x, y)
            else:
                weights = copy.copy(y)
                y = sm.add_constant(y)
                model = sm.WLS(x, y, weights=[1/w for w in weights])
                results = model.fit()
                to_return = argparse.Namespace()
                to_return.rvalue = np.sqrt(results.rsquared)
                return to_return    # short circuit this for now, below is giving me trouble
                to_return.intercept = results.params.squeeze()[0]
                to_return.slope = results.params.squeeze()[1]
                to_return.stderr = np.mean([x**2 for x in results.resid])   # technically this is mse instead
                return to_return

        def filter_data(x, y):
            # Helper function
            if settings.correlation_threshold is None:
                return x, y
            else:
                new_x = list(copy.copy(x))
                new_y = list(copy.copy(y))
                for ii in range(len(y)):
                    if y[ii] <= settings.correlation_threshold:
                        new_y.remove(y[ii])
                        new_x.remove(x[ii])
                return new_x, new_y

        if model == models[0] and settings.train_split < 1:  # do experimental model separately
            keras_model = keras.saving.load_model(model, custom_objects={"ConstantLayer": ConstantLayer})
            if len(test_labels) == 0:
                raise RuntimeError('train_split too large: test split has no members.')
            predictions = keras_model.predict(test)
            this_r = linregress(filter_data(predictions[:, 0], test_labels))
        elif model == models[-1] and settings.prosst:   # handle prosst separately
            # Load model pickle file
            model = pickle.load(open(model, 'rb'))

            # Convert mutated sequence to :-separated one-letter-code index one-letter-code format (e.g., D224G:S40G)
            mutants = []
            for seq in dataset_dict['examples']:
                this_muts = []
                mutseq = decode(seq)
                assert len(mutseq) == len(settings.wt_seq)
                for ii in range(len(mutseq) - 1):   # todo: - 1 is a kludge to fix an avgfp-specific issue, remove in the future
                    if not mutseq[ii] == settings.wt_seq[ii]:
                        this_muts.append(settings.wt_seq[ii] + str(ii + 1) + mutseq[ii])
                mutants.append(':'.join(this_muts))

            # Compute scores
            predictions = []
            for mutant in mutants:
                mutant_score = 0
                for sub_mutant in mutant.split(":"):
                    wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
                    pred = model['logits'][idx, model['vocab'][mt]] - model['logits'][idx, model['vocab'][wt]]
                    mutant_score += pred.item()
                predictions.append(mutant_score)

            # Compute correlations
            this_r = linregress(filter_data(predictions, dataset_labels[:, 0]))
        else:
            keras_model = keras.saving.load_model(model, custom_objects={"ConstantLayer": ConstantLayer})
            predictions = keras_model.predict(train)
            this_r = linregress(filter_data(predictions[:, 0], dataset_labels[:, 0]))

        correlations[label] = this_r

    with open('correlations_' + str(step) + '.out', 'w') as f:
        for label in labels:
            f.write(label + '\t' + str(correlations[label]) + '\n')

    return correlations

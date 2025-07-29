import os
import copy
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import train_score_function

def encode(examples):
    # Encode sequence with mutations as a list of integers

    ones = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P', 'A',
            'V', 'I', 'L', 'M', 'F', 'Y', 'W']

    return [[ones.index(res) for res in seq] for seq in examples]


def evaluate_score(model_predictions, correlations, previous_seqs_indices, step, settings):
    # Evaluate score for each row of model_predictions and return list of scores
    scores = []

    if settings.prediction_type == 'correlation':
        rvalues = [value.rvalue for value in list(correlations.values())]
        for ii in range(model_predictions.shape[1]):
            predictions = [(p * 2) - 1 for p in model_predictions[:, ii]]   # rescale from [0, 1] to [-1, 1]
            score = np.dot(np.array(predictions), np.array(rvalues))
            scores.append(score)
    elif settings.prediction_type == 'correlation_squared':
        rvalues = [value.rvalue for value in list(correlations.values())]
        for ii in range(model_predictions.shape[1]):
            predictions = [(p * 2) - 1 for p in model_predictions[:, ii]]   # rescale from [0, 1] to [-1, 1]
            csquares = [value * abs(value) for value in rvalues]    # x * abs(x) instead of x^2 to preserve sign
            score = np.dot(np.array(predictions), np.array(csquares))
            scores.append(score)
    elif settings.prediction_type == 'correlation_over_stderr':
        rvalues = [value.rvalue for value in list(correlations.values())]
        stderr = [value.stderr for value in list(correlations.values())]
        for ii in range(model_predictions.shape[1]):
            predictions = [(p * 2) - 1 for p in model_predictions[:, ii]]   # rescale from [0, 1] to [-1, 1]
            r_over_stderr = [rvalues[jj] / stderr[jj] for jj in range(len(rvalues))]    # x * abs(x) instead of x^2 to preserve sign
            score = np.dot(np.array(predictions), np.array(r_over_stderr))
            scores.append(score)
    elif settings.prediction_type == 'correlation_squared_weighted':
        rvalues = [value.rvalue for value in list(correlations.values())]
        for ii in range(model_predictions.shape[1]):
            weight_lines = open(settings.pearsons_r_file, 'r').readlines()
            assert len(weight_lines) == len(correlations.keys()) - 1    # -1 for experimental
            weight_lines_sorted = ['' for _ in range(len(weight_lines))]
            for line in weight_lines:
                weight_lines_sorted[list(correlations.keys()).index(line.split()[0] + '.keras') - 1] = line
            weight_lines = weight_lines_sorted
            for jj in range(len(correlations.keys()) - 1):
                assert weight_lines[jj].split()[0] + '.keras' == list(correlations.keys())[jj + 1]
            weights = [weight.split()[1].replace('PearsonRResult(statistic=', '').replace(',', '') for weight in weight_lines]
            for qq in range(len(weights)):
                if weights[qq] == 'nan':
                    weights[qq] = '0'
            weights = [float(weight) for weight in weights]
            predictions = [(p * 2) - 1 for p in model_predictions[:, ii]]   # rescale from [0, 1] to [-1, 1]
            csquares = [value * abs(value) for value in rvalues]    # x * abs(x) instead of x^2 to preserve sign
            weighted_csquares = [csquares[0]] + [csquares[kk+1] * weights[kk] for kk in range(len(weights))]
            score = np.dot(np.array(predictions), np.array(weighted_csquares))
            scores.append(score)
    elif settings.prediction_type == 'correlation_cubed':
        rvalues = [value.rvalue for value in list(correlations.values())]
        for ii in range(model_predictions.shape[1]):
            predictions = [(p * 2) - 1 for p in model_predictions[:, ii]]   # rescale from [0, 1] to [-1, 1]
            csquares = [value * abs(value) * abs(value) for value in rvalues]    # x * abs(x) instead of x^2 to preserve sign
            score = np.dot(np.array(predictions), np.array(csquares))
            scores.append(score)
    elif settings.prediction_type == 'regression':
        for ii in range(model_predictions.shape[1]):
            predictions = model_predictions[:, ii]
            rs = list(correlations.values())
            assert len(predictions) == len(rs)
            score = sum([(rs[jj].intercept + rs[jj].slope * predictions[jj]) / (rs[jj].stderr)**2 for jj in range(len(rs))])
            scores.append(score)
    elif settings.prediction_type == 'regression_lowpass':
        def lowpass(x):
            if x < 0:
                return x
            else:
                return 0
        for ii in range(model_predictions.shape[1]):
            predictions = model_predictions[:, ii]
            rs = list(correlations.values())
            assert len(predictions) == len(rs)
            score = sum([lowpass((rs[jj].intercept + rs[jj].slope * predictions[jj])) / (rs[jj].stderr) ** 2 for jj in
                         range(len(rs))])
            scores.append(score)
    elif settings.prediction_type == 'correlation_top_half':
        rvalues = [value.rvalue for value in list(correlations.values())]
        for ii in range(model_predictions.shape[1]):
            predictions = [(p * 2) - 1 for p in model_predictions[:, ii]]   # rescale from [0, 1] to [-1, 1]
            filtered_correlations = [0 for _ in rvalues]
            median_correlation = np.percentile([abs(item) for item in rvalues], 90)
            for jj in range(len(filtered_correlations)):
                if [abs(item) for item in rvalues][jj] >= median_correlation:
                    filtered_correlations[jj] = rvalues[jj]
            score = np.dot(np.array(predictions), np.array(filtered_correlations))
            scores.append(score)
    elif settings.prediction_type == 'learned':
        # Put together training dataset from previous seqs;
        # examples in model_predictions[:, previous_seqs_indices];
        # labels in that same order in settings.output_filename
        labels = ' '.join([' '.join(line.split()[1:]) for line in open(settings.output_filename, 'r').readlines()[1:]])
        labels = np.array([float(item) for item in labels.split()])
        assert len(labels) == len(previous_seqs_indices)
        examples = np.transpose(model_predictions[:, previous_seqs_indices])
        dataset = {'examples': examples, 'labels': labels}
        pickle.dump(dataset, open('learned_score_function_trainset_' + str(step) + '.pkl', 'wb'))

        if not os.path.exists('learned_score_function_' + str(step) + '.keras'):
            model = train_score_function.train(dataset, step, settings)
        else:
            model = keras.saving.load_model('learned_score_function_' + str(step) + '.keras')

        ii = 0
        while ii < model_predictions.shape[1]:
            predictions = np.transpose(model_predictions[:, ii:ii + 8])
            batched_predictions = tf.data.Dataset.from_tensor_slices((predictions, [0 for _ in range(len(predictions))]))
            batched_predictions = batched_predictions.batch(8)
            scores += list(model.predict(batched_predictions)[:].squeeze())
            ii += 8

    else:
        raise RuntimeError('Unrecognized prediction_type: ' + settings.prediction_type)

    return list(np.nan_to_num(scores))


def from_file(models, correlations, seqfile, step, settings):
    # Evaluate candidate sequences in a file
    candidates = [line.split()[0] for line in open(settings.sequences_file, 'r').readlines()]
    encoded_candidates = encode(candidates)
    model_predictions = np.zeros(shape=(len(models), len(candidates)))
    model_indices = range(len(models))

    # Extract list of previous sequences from seqfile
    previous_seqs = [line.strip('\n') for line in open(seqfile, 'r').readlines()]
    previous_seqs_indices = [candidates.index(seq) for seq in previous_seqs]
    assert len(previous_seqs) == len(set(previous_seqs))    # if not all unique, something has gone wrong

    # Sloppy way to work out how to interpret contents of existing predictions file, if necessary
    if os.path.exists(settings.predictions_file + '/model_predictions.pkl') and settings.reuse_predictions:
        model_predictions = pickle.load(open(settings.predictions_file + '/model_predictions.pkl', 'rb'))
        try:
            assert model_predictions.shape[0] == len(models)
        except AssertionError:
            if model_predictions.shape[0] - 1 == len(models) and settings.skip_experimental_model:
                print('Warning: loaded predictions file ' + settings.predictions_file + '/model_predictions.pkl, but '
                      'found it to contain one more predictions than this run has assigned models. Assuming that this '
                      'is because of skip_experimental_model = True for this run and continuing.')
                model_predictions = model_predictions[1:,:]
            elif model_predictions.shape[0] - 1 == len(models) and not settings.skip_experimental_model:
                print('Warning: loaded predictions file ' + settings.predictions_file + '/model_predictions.pkl, but '
                      'found it to contain one more predictions than this run has assigned models. Assuming that this '
                      'is because skip_experimental_model was False when that file was created and continuing.')
                #model_predictions = model_predictions
                raise RuntimeError('Actually, that\'s not supported yet!')  # todo: implement
            if model_predictions.shape[0] + 1 == len(models) and settings.prosst:
                print('Warning: loaded predictions file ' + settings.predictions_file + '/model_predictions.pkl, but '
                      'found it to contain two more predictions than this run has assigned models. Assuming that this '
                      'is because prosst is set for this run and continuing.')
                model_predictions = np.concatenate((model_predictions, np.zeros(shape=(1, len(candidates)))))
            else:
                raise RuntimeError('Tried to load ' + settings.predictions_file + '/model_predictions.pkl, but found it'
                                   ' to contain a different number of predictions than this run has assigned models!')
        if not settings.skip_experimental_model:
            model_indices = [0]     # need to still redo the experimental model
        else:
            model_indices = []

    for ii in model_indices:
        model = models[ii]

        if model == models[-1] and settings.prosst:   # handle prosst separately
            # Load model pickle file
            model = pickle.load(open(model, 'rb'))

            if settings.model_architecture == 'avgfp' and len(settings.wt_seq) == 227:  # kludge to fix an avgfp-specific issue, remove in the future
                offset = 1
            else:
                offset = 0

            # Convert mutated sequence to :-separated one-letter-code index one-letter-code format (e.g., D224G:S40G)
            if not os.path.exists('prosst_mutants.pkl'):
                mutants = []
                for seq in candidates:
                    this_muts = []
                    mutseq = seq
                    assert len(mutseq) == len(settings.wt_seq)
                    for jj in range(len(mutseq) - offset):
                        if not mutseq[jj] == settings.wt_seq[jj]:
                            this_muts.append(settings.wt_seq[jj] + str(jj + 1) + mutseq[jj])
                    mutants.append(':'.join(this_muts))
                pickle.dump(mutants, open('prosst_mutants.pkl', 'wb'))
            else:
                mutants = pickle.load(open('prosst_mutants.pkl', 'rb'))

            # Compute scores
            logits = model['logits']
            vocab = model['vocab']
            predictions = []
            for mutant in mutants:
                mutant_score = 0
                for sub_mutant in mutant.split(":"):
                    wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
                    pred = logits[0, idx, vocab[mt]] - logits[0, idx, vocab[wt]]
                    mutant_score += pred.item()
                predictions.append(mutant_score)

            model_predictions[ii, :] = np.array(predictions)
            continue

        loaded_model = keras.saving.load_model(model, safe_mode=False)

        jj = 0
        while jj < len(candidates):
            encoded_seqs = [np.array(c) for c in encoded_candidates[jj:jj+settings.batch_size]]
            batched_seqs = tf.data.Dataset.from_tensor_slices((encoded_seqs, [0 for _ in range(len(encoded_seqs))]))
            batched_seqs = batched_seqs.batch(settings.batch_size)
            predictions = loaded_model.predict(batched_seqs)[:].squeeze()
            model_predictions[ii, jj:jj+settings.batch_size] = predictions
            jj += settings.batch_size

    assert model_predictions.shape[0] == len(models)
    assert model_predictions.shape[1] == len(candidates)

    if not os.path.exists('model_predictions.pkl'):
        pickle.dump(model_predictions, open('model_predictions.pkl', 'wb'))

    def safediv(a, b):
        # Helper function
        if b == 0:
            return 0
        elif np.isnan(a) or np.isnan(b):
            return 0
        else:
            return a / b

    if settings.compare_prediction_types:
        for prediction_type in settings.compare_prediction_types:
            temp_settings = copy.copy(settings)
            temp_settings.prediction_type = prediction_type
            scores = evaluate_score(model_predictions, correlations, previous_seqs_indices, step, temp_settings)
            try:
                labels = [line.split()[1] for line in open(settings.sequences_file, 'r').readlines()]
                float(labels[0])    # check that we can cast to float, i.e., that this is a number
            except (IndexError, ValueError):
                raise RuntimeError('Use of compare_prediction_types requires numerical labels to be present in the sequences_file')
            with open(prediction_type + '_' + str(step) + '_benchmark.out', 'w') as f:
                f.write('\n'.join([str(scores[ii]) + '\t' + labels[ii] for ii in range(len(scores))]))

    scores = evaluate_score(model_predictions, correlations, previous_seqs_indices, step, settings)
    output_seqs = []

    if settings.selection_method == 'top':
        while len(output_seqs) < settings.step_size:    # ugly approach to selecting top sequences based on max scores
            jj = scores.index(max(scores))

            # Add seq to output_seqs if and only if it's not already been selected in a previous round
            if not candidates[jj] in previous_seqs:
                output_seqs.append(candidates[jj])

            scores.remove(scores[jj])
            candidates.remove(candidates[jj])
    elif settings.selection_method == 'probabilistic':
        # Remove candidates previously selected
        for seq in previous_seqs:
            jj = candidates.index(seq)
            candidates.remove(candidates[jj])
            scores.remove(scores[jj])

        scores = [score ** 4 for score in scores]   # to bias strongly towards high-scorers
        if sum(scores) == 0:
            scores = [1 for _ in scores]

        output_seqs = np.random.choice(candidates, size=settings.step_size, replace=False,
                                       p=[safediv(score, sum(scores)) for score in scores])
    else:
        raise RuntimeError('Unrecognized selection_method: ' + settings.selection_method)

    assert not any([seq in previous_seqs for seq in output_seqs])

    return output_seqs


def from_scratch(models, correlations, seqfile, step, settings):
    # Evaluate randomly generated candidate sequences
    raise RuntimeError('from_scratch not yet implemented')  # todo: implement from_scratch


def main(models, correlations, step, seqfile, settings):
    if settings.campaign == 'from_file':
        seqs = from_file(models, correlations, seqfile, step, settings)
    elif settings.campaign == 'from_scratch':
        seqs = from_scratch(models, correlations, seqfile, step, settings)
    else:
        raise RuntimeError('Unrecognized campaign: ' + settings.campaign)

    with open('seqs_' + str(step) + '.out', 'w') as f:
        for seq in seqs:
            f.write(seq + '\n')

    return 'seqs_' + str(step) + '.out'

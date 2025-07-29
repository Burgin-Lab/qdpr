import os
import sys
import numpy as np
import shutil
import pickle
import utilities
import itertools
import train_experimental
import select_sequences
from configure import configure

def main(settings):
    if os.path.exists(settings.working_directory):
        if not settings.overwrite:
            raise RuntimeError('Working directory ' + settings.working_directory + ' already exists and '
                               'overwrite = False')
        else:
            shutil.rmtree(settings.working_directory)
    os.mkdir(settings.working_directory)    # create working directory
    shutil.copy(settings.sequences_file, settings.working_directory)
    shutil.copy(settings.initial_dataset_file, settings.working_directory)
    os.chdir(settings.working_directory)    # move to working directory

    if settings.initial_dataset_type == 'random':
        seqfile = None
    elif settings.initial_dataset_type == 'file':
        seqfile = settings.initial_dataset_file
    else:
        raise RuntimeError('Unrecognized initial_dataset_type: ' + settings.initial_dataset_type)

    full_dataset, seqs = utilities.make_dataset('initial_dataset', seqfile, settings)    # get initial dataset
    with open('all_seqs_-1.out', 'w') as f:
        for seq in seqs:
            f.write(str(seq) + '\n')
    seqfile = 'all_seqs_-1.out'
    utilities.update_output(full_dataset, -1, settings)    # write labels to output file

    pre_models = utilities.get_models(settings)     # load specified pre-trained models

    for step in range(settings.n_steps):    # loop through n_steps rounds of selection
        # First, if settings.campaign == 'random', do that
        if settings.campaign == 'random_from_file':
            if settings.rng_seed < 0:
                settings.rng_seed = None
            rng = np.random.default_rng(settings.rng_seed)
            settings.rng_seed = rng.integers(99999)     # use rng seed to choose new rng seed
            this_dataset, _ = utilities.make_dataset(str(step), None, settings)
        else:
            if not settings.skip_experimental_model:
                # Train experimental backbone on available data
                exp_model = train_experimental.train(full_dataset, step, pre_models, settings)
                models = list(itertools.chain.from_iterable([[exp_model], pre_models]))
            else:
                models = pre_models

            # Generate correlations for each model on current dataset
            correlations = utilities.generate_correlations(full_dataset, models, step, settings)

            # Select sequences for next step
            seqs = select_sequences.main(models, correlations, step, seqfile, settings)
            assert len(open(seqs, 'r').readlines()) == settings.step_size

            # Exit if campaign is from_scratch, since we have no basis on which to continue
            if settings.campaign == 'from_scratch':
                print('Selected sequences written to output file: ' + seqs)
                break

            # Make a dataset for this step and update full_dataset
            this_dataset, _ = utilities.make_dataset(str(step), seqs, settings)
            assert len(pickle.load(open(this_dataset, 'rb'))['examples']) == settings.step_size
            with open('all_seqs_' + str(step) + '.out', 'w') as f:
                for line in open(seqfile, 'r').readlines():
                    f.write(line)
                for line in open(seqs, 'r').readlines():
                    f.write(line)
            seqfile = 'all_seqs_' + str(step) + '.out'
            full_dataset, _ = utilities.make_dataset(str(step) + '_full', 'all_seqs_' + str(step) + '.out', settings)

        # Produce output and prepare for next step, if applicable
        utilities.update_output(this_dataset, step, settings)


if __name__ == "__main__":
    if not os.path.exists(sys.argv[1]):
        raise RuntimeError('Cannot find config file: ' + sys.argv[1])
    settings = configure(sys.argv[1])

    try:
        settings.working_directory = sys.argv[2]
    except IndexError:
        pass

    main(settings)
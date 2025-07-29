import tensorflow as tf
from tensorflow import keras

def get_datasets(dataset, settings):
    train = tf.data.Dataset.from_tensor_slices((dataset['examples'], dataset['labels']))
    return train


def run_experiment(model, loss, train_dataset):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=loss,
        metrics=[keras.metrics.MeanSquaredError()],
    )

    history = model.fit(train_dataset, epochs=400)
    _, rmse = model.evaluate(train_dataset, verbose=0)
    return history


def create_model(settings):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU()),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    return model


def train(dataset, step, settings):
    training_index = 0
    train_dataset = get_datasets(dataset, settings)

    BATCH_SIZE = 8

    model = create_model(settings)

    SHUFFLE_BUFFER_SIZE = 100
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    run_experiment(model, keras.losses.MeanSquaredError(), train_dataset)

    model.save('learned_score_function_' + str(step) + '.keras')
    return model

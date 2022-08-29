import tensorflow as tf
import numpy as np
import util
import os


def vectorize_string(string):
    return np.array([char2id[c] for c in string])

def get_batch(vectorized_songs, seq_length, batch_size):
    n = vectorized_songs.shape[0] - 1
    ids = np.random.choice(n - seq_length, batch_size)
    input_batch = [vectorized_songs[i : i + seq_length] for i in ids]
    output_batch = [vectorized_songs[i + 1 : i + 1 + seq_length] for i in ids]
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch

def generate_text(model, start_string, generation_length=1000):
    input_eval = [char2id[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()

    for i in range(generation_length):
        preds = model(input_eval)
        preds = tf.squeeze(preds, 0)
        predicted_id = tf.random.categorical(preds,
                                             num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(id2char[predicted_id])

    return start_string + ''.join(text_generated)

def LSTM(rnn_units):
    return tf.keras.layers.LSTM(rnn_units,
                                return_sequences=True,
                                recurrent_initializer='glorot_uniform',
                                recurrent_activation='sigmoid',
                                stateful=True)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,
                                  embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        LSTM(rnn_units),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                           logits,
                                                           from_logits=True)
    return loss

@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = compute_loss(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

songs = util.load_training_data()
songs_joined = '\n\n'.join(songs)
vocab = sorted(set(songs_joined))
char2id = {s: i for i, s in enumerate(vocab)}
id2char = np.array(vocab)
vectorized_songs = vectorize_string(songs_joined)
vocab_size, embedding_dim, rnn_units, batch_size = len(vocab), 256, 1024, 4
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
learning_rate = 1e-2
optimizer = tf.keras.optimizers.Adam(learning_rate)
num_training_iterations = 2000
seq_length = 25
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'my_ckpt')
history = []

# for i in range(num_training_iterations + 1):
#     print(i)
#     x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
#     loss = train_step(model, x_batch, y_batch)
#     history.append(loss.numpy().mean())
#     if i % seq_length == 0:
#         model.save_weights(checkpoint_prefix)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

bach_original = "X: 1\n\
T: Bach BWV 944\n\
M: C\n\
L: 1/8\n\
Z: Contributed 2017-05-20 08:28:11 by tipto b1041397\n\
K: Am\n\
EGce ecGE | DFBd dBFD E^GBd dBGE | F^GBd dBGF EGAc cAGE | DFAc cAFD DF^GB BGFD | DF^GB BGFD CEA AEC |\n\
G_Beg geBG GA^ceg gecAG | G_B^ceg gecBG Adf fdA | ^GBdf fdBG GAce ecAG | zAce ecA ^G2 ^FAB^d dBAF | E^GBe eBGHE |]"

util.play_song(original_bach, save='bach_original')

bach_test = "X: 1\n\
T: Bach BWV 944\n\
M: C\n\
L: 1/8\n\
Z: Contributed 2017-05-20 08:28:11 by tipto\n\
K: Am\n\
EGce ecGE | DFBd dBFD E^GBd dBGE"

util.play_generated_song(generate_text(model, start_string=bach_test))

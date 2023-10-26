import tensorflow as tf
import numpy as np
import os
import pandas as pd

path = "C:/anime/anime_names.csv"

names_df = pd.read_csv(path)
names_ls = list(names_df["0"])
print("Number of names: ",len(names_ls))

names_df.head()

text = ""
for name in names_ls:
    text+=name
    
unique_chars = list(set(text))
num_chars = len(unique_chars)
print("Number of unique characters: ",num_chars)

char_to_id = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_chars)
id_to_char = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=char_to_id.get_vocabulary(), invert=True)

char_ls = tf.strings.unicode_split(names_ls,input_encoding="UTF-8")
name_byte_ls = list(tf.strings.reduce_join(char_ls, axis=-1))
id_ls = list(char_to_id(char_ls))
all_ids = char_to_id(tf.strings.unicode_split(text, 'UTF-8'))
print("Example name: ",name_byte_ls[0])
print("Split: ",char_ls[0])
print("It's encoding: ",id_ls[0])

print("Number of chars in all names: ",tf.shape(all_ids)[0])

len(char_to_id.get_vocabulary())
padded_id = tf.keras.preprocessing.sequence.pad_sequences(id_ls,padding = "post")
print("Padded ids shape: ",padded_id.shape)
seq_length = padded_id.shape[1]
BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE
SHUFFLE = 60000
ids_dataset = tf.data.Dataset.from_tensor_slices(padded_id)

def io_split(id_arr):
    inp = id_arr[:-1]
    out = id_arr[1:]
    return inp,out
ids_dataset = ids_dataset.map(io_split)
ids_dataset = ids_dataset.shuffle(SHUFFLE).batch(BATCH_SIZE,drop_remainder=True).prefetch(AUTOTUNE)

ids_dataset
print(char_to_id.get_vocabulary())
rnn_units = 1024
embedding_dim = 256

class GenerateModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True, 
                                   return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else: 
            return x
        
model = GenerateModel(vocab_size=len(char_to_id.get_vocabulary()),embedding_dim=embedding_dim,rnn_units=rnn_units)


for input_batch, target_batch in ids_dataset.take(1):
    batch_pred = model(input_batch)
    print(batch_pred.shape)
    
print("MODEL SUMMARY")
model.summary()

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
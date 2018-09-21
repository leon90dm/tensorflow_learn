from tensorflow import keras

imdb = keras.datasets.imdb

# The dataset comes preprocessed: each example is an array
# of integers representing the words of the movie review.
# Each label is an integer value of either 0 or 1,
# where 0 is a negative review, and 1 is a positive review.
# num_words=10000 保留最常用的10,000个单词
vocab_size = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print("{} : {}".format(decode_review(train_data[0]), train_labels[0]))

#
train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=256, padding="post",
                                                        value=word_index["<PAD>"])
test_data = keras.preprocessing.sequence.pad_sequences(test_data, maxlen=256, padding="post",
                                                       value=word_index["<PAD>"])
# building model

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation=keras.activations.relu),
    keras.layers.Dense(1, activation=keras.activations.sigmoid)
])

model.summary()

model.compile(keras.optimizers.Adam(),
              loss=keras.losses.binary_crossentropy,
              metrics=["accuracy"])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(x=partial_x_train, y=partial_y_train,
                    epochs=40, batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


results = model.evaluate(test_data, test_labels)

print(results)
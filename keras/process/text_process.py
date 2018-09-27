import keras


def text_to_words():
    with open('text.txt') as f:
        for line in f.readlines():
            out = keras.preprocessing.text.text_to_word_sequence(line,
                                                                 filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                                                                 lower=True, split=' ')
            print(out)


hot = keras.preprocessing.text.one_hot("machine learning save", n=5, split=' ')
print(hot)

"""
Bag of Embeddings model
Reference : http://www.aclweb.org/anthology/P/P16/P16-1145.pdf

authors:
@hadyelsahar    hadyelsahar@gmail.com

"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalAveragePooling1D
from datautils.datagenerator import WikiReadingDataGenerator

TOP_WORDS = 50000
MAX_SEQ_LENGTH = 300
EMB_VEC_LENGTH = 50
N_EPOCHS = 10
BATCH_SIZE = 32
W2V = False

trainfiles = ['./data/train-00000-of-00150.json']
testfiles = ['./data/test-00000-of-00015.json']

g = WikiReadingDataGenerator(x_dict='./vocab/document.vocab', y_dict='./vocab/raw_answer.vocab')

generator_function = g.generate(trainfiles, 1, nb_words=TOP_WORDS, maxlen=MAX_SEQ_LENGTH,
                                x_vectorize_function=g.bow_x_vectorizer, y_vectorize_function=g.onehot_y_vectorizer,
                                oov_id=1, pad_id=0)

model = Sequential()
model.add(Embedding(TOP_WORDS, EMB_VEC_LENGTH, input_length=MAX_SEQ_LENGTH))
model.add(GlobalAveragePooling1D())

model.add(Dense(output_dim=g.y_onehot_encoder.shape, input_dim=MAX_SEQ_LENGTH))
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

print(model.summary())


model.fit_generator(generator_function, samples_per_epoch=1000, nb_epoch=4)

generator_function = g.generate(testfiles, 128, nb_words=50000,
                                x_vectorize_function=g.bow_x_vectorizer, y_vectorize_function=g.onehot_y_vectorizer,
                                oov_id=1, pad_id=0)

scores = model.evaluate_generator(generator_function, 100, nb_worker=1, pickle_safe=False)











import numpy as np
import keras
from keras.layers import Dense, Input, Flatten, Dropout
from keras.models import Model
from game import Game


def get_model() -> keras.Model:
    """ Building a model that predicts """
    input_layer = Input(shape=(8, 8, 3, ))
    flattened = Flatten()(input_layer)
    flattened = Dense(256, activation='relu')(flattened)
    flattened = Dense(64, activation='relu')(flattened)
    output_layer = Dense(1, activation='sigmoid')(flattened)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    # print(model.summary())
    return model


def train_on_batch(model, x, y) -> keras.Model:
    """ Function to train model on data of one game"""
    model.fit(x, y, batch_size=None, epochs=1, verbose=0)
    return model


def get_prediction(model, x) -> np.array:
    return model.predict(x)






if __name__ == "__main__":
    model = get_model()
    game = Game(size=8, output='emoji')
    test_batch, winner = game.play()
    print(test_batch[0].shape, len(test_batch))
    print("TEST WINNER: ", winner)
    test_batch = np.stack(test_batch, axis=0)
    print(test_batch.shape)
    # y = np.full((1, test_batch.shape[0]), fill_value=winner, dtype=int)
    # print(y)
    # print(test_batch[:1,])
    # for i in test_batch[:3,:4, :6, :]:
    #     print(i, end='\n')

    model = train_on_batch(model=model, x=test_batch, y=np.full((test_batch.shape[0], 1), fill_value=winner, dtype=int))
    before = get_prediction(model=model, x=test_batch)
    print(before[0][0])
    #
    #
    for _ in range(0):
        game = Game(size=8, output='emoji')
        train_batch, winner = game.play()
        train_batch = np.stack(train_batch, axis=0)
        # print(train_batch)
        y = np.full((train_batch.shape[0], 1), fill_value=winner, dtype=int)
        model = train_on_batch(model=model, x=train_batch, y=y)

    # get_prediction(model=model, x=test_batch[4])
    after = get_prediction(model=model, x=test_batch)
    print(np.stack((before, after, after-before), axis=2))

    for i in test_batch[:3,]:
        # print(i.shape)
        print('PROBA:  ', get_prediction(model=model, x=np.array((i,)))[0][0], end='\n')





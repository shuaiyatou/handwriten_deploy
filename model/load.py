from keras.models import model_from_json
import tensorflow as tf


def init_model():
    with open('model/model.json') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model/model.h5')
    print('Loaded Model from disk!')

    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
    print()
    graph = tf.get_default_graph()
    return loaded_model, graph
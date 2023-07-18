import tensorflow as tf
import kdens
from dataclasses import dataclass
import json

with open(f"./voc.json", 'r') as inp:
  voc = json.load(inp)

@dataclass
class KDESolConfig:
    vocab_size: int = len(voc)
    batch_size: int = 16
    buffer_size: int = 10000
    rnn_units: int = 64
    hidden_dim: int = 32
    embedding_dim: int = 64
    reg_strength: float = 0.01
    lr: float = 1e-4
    drop_rate: float = 0.35
    nmodels: int = 8
    adv_epsilon: float = 1e-3
    epochs: int = 150

class KDESol:
    def __init__(self, config, weigths_path=None):
        self.model = self.create_model(config)
        self.config = config
        self.voc = voc
        if weigths_path:
            self.model.load_weights(weigths_path)

    def load_weights(self, model_path):
        models = []
        for i in range(self.config.nmodels):
            with open(f"{model_path}/m{i}.json", "r") as json_file:
                json_model = json_file.read()
                m = tf.keras.models.model_from_json(json_model)
                m.load_weights(f"{model_path}/m{i}.h5")
            
                models.append(m)
        m = kdens.DeepEnsemble(self.create_inf_model, 
                                self.config.nmodels, 
                                self.config.adv_epsilon)
        m.models = models
        self.model = m

    def create_inf_model(self, config):
        inputs = tf.keras.Input(shape=(None,))

        # make embedding and indicate that 0 should be treated as padding mask
        e = tf.keras.layers.Embedding(input_dim=config.vocab_size, 
                                            output_dim=config.embedding_dim,
                                            mask_zero=True)(inputs)

        # RNN layer
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.rnn_units, return_sequences=True))(e)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.rnn_units))(x)
        x = tf.keras.layers.LayerNormalization()(x)
        # a dense hidden layer
        x = tf.keras.layers.Dense(config.hidden_dim, activation="swish")(x)
        x = tf.keras.layers.Dense(config.hidden_dim // 2, activation="swish")(x)
        # predicting prob, so no activation
        muhat = tf.keras.layers.Dense(1)(x)
        stdhat = tf.keras.layers.Dense(1, activation='softplus')(x)
        out = tf.squeeze(tf.stack([muhat, stdhat], axis=-1))
        model = tf.keras.Model(inputs=inputs, outputs=out, name='sol-rnn-infer')
        return model

    def create_model(self, config):
        inputs = tf.keras.Input(shape=(None,))

        # make embedding and indicate that 0 should be treated as padding mask
        e = tf.keras.layers.Embedding(input_dim=config.vocab_size, 
                                            output_dim=config.embedding_dim,
                                            mask_zero=True)(inputs)
        e = tf.keras.layers.Dropout(config.drop_rate)(e)
        # RNN layer
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.rnn_units, return_sequences=True,  kernel_regularizer='l2'))(e)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.rnn_units, kernel_regularizer='l2'))(x)
        x = tf.keras.layers.LayerNormalization()(x)
        # a dense hidden layer
        x = tf.keras.layers.Dense(config.hidden_dim, activation="swish",  kernel_regularizer='l2')(x)
        x = tf.keras.layers.Dropout(config.drop_rate)(x)
        x = tf.keras.layers.Dense(config.hidden_dim // 2, activation="swish",  kernel_regularizer='l2')(x)
        x = tf.keras.layers.Dropout(config.drop_rate)(x)
        # predicting prob, so no activation
        muhat = tf.keras.layers.Dense(1)(x)
        stdhat = tf.keras.layers.Dense(1, 
                                    activation='softplus', 
                                    bias_constraint=tf.keras.constraints.MinMaxNorm( 
                                        min_value=1e-6, max_value=1000.0, rate=1.0, axis=0))(x)
        out = tf.squeeze(tf.stack([muhat, stdhat], axis=-1))
        model = tf.keras.Model(inputs=inputs, outputs=out, name='sol-rnn')
        partial_in = tf.keras.Model(inputs=inputs, outputs=e)
        partial_out = tf.keras.Model(inputs=e, outputs=out)
        return model, partial_in, partial_out

    def run(self, x):
        return self.model(x)
    
    def __call__(self, x):
        return self.run(x)

    def get_config(self):
        pass


    
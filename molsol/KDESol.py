import os
import tensorflow as tf
import kdens
from dataclasses import dataclass
import json
import selfies as sf
import numpy as np

model_path = os.path.abspath(
    os.path.join(os.path.split(__file__)[0], "kde10_lstm_r")
                 )
vocab_path = os.path.abspath(
    os.path.join(os.path.split(__file__)[0], "voc.json")
                 )

with open(vocab_path, 'r') as inp:
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
    nmodels: int = 10
    adv_epsilon: float = 1e-3
    epochs: int = 150

class KDESol:
    def __init__(self, config=KDESolConfig(), weigths_path=model_path):
        self.config = config
        self.voc = voc
        if weigths_path:
            self.load_model(weigths_path)
        else:
            print("Weights not found. Creating new DNN model.")
            self.model = self.create_model()

    def load_model(self, model_path):
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

    def create_inf_model(self):
        config = self.config

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

    def create_model(self):
        config = self.config

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
        x = self.stoi(self.encoder(x))
        x = np.array([x, x])
        return self.model.predict(x)[0]
    
    def __call__(self, x):
        return self.run(x)

    def get_config(self):
        pass

    def encoder(self, smiles):
        try:
            return sf.encoder(smiles)
        except:
            return None

    def stoi(self, selfies):
        try:
            label, one_hot = sf.selfies_to_encoding(
            selfies=selfies,
            vocab_stoi=voc,
            pad_to_len=634,
            enc_type="both"
            )
        except:
            return None
        return label


if __name__ == "__main__":
    import pandas as pd
    
    m = KDESol(KDESolConfig(), model_path)
    print(m.model)
    
    df = pd.read_csv("AqSolDB.csv", sep='\t', nrows=100)
    df = df[['SMILES', 'Solubility']]
    df["SELFIES"] = df["SMILES"].map(m.encoder)
    print(df['SELFIES'].iloc[0:1])
    print(m(df['SELFIES'].iloc[0:1]))

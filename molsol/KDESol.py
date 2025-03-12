import os
import tensorflow as tf
import keras
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

@keras.utils.register_keras_serializable()
class StackLayer(keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A Stack layer should be called on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape != list(shape):
                raise ValueError('All inputs must have the same shape.')
        output_shape.insert(self.axis, len(input_shapes))
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

@keras.utils.register_keras_serializable()
class SqueezeLayer(keras.layers.Layer):
    def __init__(self, axis=None, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if self.axis is None:
            return tuple(d for d in input_shape if d != 1)
        else:
            if isinstance(self.axis, int):
                del input_shape[self.axis]
            else:
                for idx in sorted(self.axis, reverse=True):
                    del input_shape[idx]
            return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

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
        custom_objects = {
            'StackLayer': StackLayer,
            'SqueezeLayer': SqueezeLayer,
            'kdens>DeepEnsemble': kdens.DeepEnsemble,
            'kdens>neg_ll': kdens.neg_ll
        }
        
        for i in range(self.config.nmodels):
            with open(f"{model_path}/m{i}.json", "r") as json_file:
                json_model = json_file.read()
                # Create a new model with the same architecture
                m = self.create_inf_model()
                # Load the weights
                m.load_weights(f"{model_path}/m{i}.h5")
                models.append(m)
                
        m = kdens.DeepEnsemble(self.create_inf_model, 
                                self.config.nmodels, 
                                self.config.adv_epsilon)
        m.models = models
        self.model = m

    def create_inf_model(self):
        config = self.config

        inputs = keras.Input(shape=(None,))

        # make embedding and indicate that 0 should be treated as padding mask
        e = keras.layers.Embedding(input_dim=config.vocab_size, 
                                            output_dim=config.embedding_dim,
                                            mask_zero=True)(inputs)

        # RNN layer
        x = keras.layers.Bidirectional(keras.layers.LSTM(config.rnn_units, return_sequences=True))(e)
        x = keras.layers.Bidirectional(keras.layers.LSTM(config.rnn_units))(x)
        x = keras.layers.LayerNormalization()(x)
        # a dense hidden layer
        x = keras.layers.Dense(config.hidden_dim, activation="swish")(x)
        x = keras.layers.Dense(config.hidden_dim // 2, activation="swish")(x)
        # predicting prob, so no activation
        muhat = keras.layers.Dense(1)(x)
        stdhat = keras.layers.Dense(1, activation='softplus')(x)
        
        # Replace tf operations with custom layers
        out = StackLayer(axis=-1)([muhat, stdhat])
        out = SqueezeLayer()(out)
        
        model = keras.Model(inputs=inputs, outputs=out, name='sol-rnn-infer')
        return model

    def create_model(self):
        config = self.config

        inputs = keras.Input(shape=(None,))

        # make embedding and indicate that 0 should be treated as padding mask
        e = keras.layers.Embedding(input_dim=config.vocab_size, 
                                            output_dim=config.embedding_dim,
                                            mask_zero=True)(inputs)
        e = keras.layers.Dropout(config.drop_rate)(e)
        # RNN layer
        x = keras.layers.Bidirectional(keras.layers.LSTM(config.rnn_units, return_sequences=True,  kernel_regularizer='l2'))(e)
        x = keras.layers.Bidirectional(keras.layers.LSTM(config.rnn_units, kernel_regularizer='l2'))(x)
        x = keras.layers.LayerNormalization()(x)
        # a dense hidden layer
        x = keras.layers.Dense(config.hidden_dim, activation="swish",  kernel_regularizer='l2')(x)
        x = keras.layers.Dropout(config.drop_rate)(x)
        x = keras.layers.Dense(config.hidden_dim // 2, activation="swish",  kernel_regularizer='l2')(x)
        x = keras.layers.Dropout(config.drop_rate)(x)
        # predicting prob, so no activation
        muhat = keras.layers.Dense(1)(x)
        stdhat = keras.layers.Dense(1, 
                                    activation='softplus', 
                                    bias_constraint=keras.constraints.MinMaxNorm( 
                                        min_value=1e-6, max_value=1000.0, rate=1.0, axis=0))(x)
        
        # Replace tf operations with custom layers
        out = StackLayer(axis=-1)([muhat, stdhat])
        out = SqueezeLayer()(out)
        
        model = keras.Model(inputs=inputs, outputs=out, name='sol-rnn')
        partial_in = keras.Model(inputs=inputs, outputs=e)
        partial_out = keras.Model(inputs=e, outputs=out)
        return model, partial_in, partial_out

    def run(self, x):
        x = self.stoi(self.encoder(x))
        if x is None:
            return None
        x = np.array([x, x])
        # Use predict instead of __call__ to avoid training mode issues
        predictions = []
        for model in self.model.models:
            pred = model.predict(x, verbose=0)
            predictions.append(pred)
        # Stack predictions and compute statistics
        predictions = np.stack(predictions, axis=0)  # [n_models, batch_size, 2]
        mean_pred = np.mean(predictions, axis=0)[0]  # [2]
        std_pred = np.std(predictions, axis=0)[0]  # [2]
        # Return mean, std, and epistemic uncertainty
        return np.array([mean_pred[0], mean_pred[1], std_pred[0]], dtype=np.float32)
    
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

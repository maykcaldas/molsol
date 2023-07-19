import sys
sys.path.append(".")

import pytest
import os


import pandas as pd
import numpy as np
from molsol import KDESol, KDESolConfig
import kdens
import keras

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

@pytest.fixture
def vocab_path():
    return os.path.abspath(
                    os.path.join(os.path.split(__file__)[0], "voc.json")
                 )

@pytest.fixture
def weigths_path():
    return os.path.abspath(
                    os.path.join(os.path.split(__file__)[0], "kde10_lstm_r")
                 )

class TestKDESol:
    
    def test_load_model_from_weigths(self):
        """
        Test to cover loading a model from a file
        """
        # Create instance of KDESol
        model = KDESol()
        # Check that model was loaded
        assert model.model is not None
        assert isinstance(model.model, kdens.kdens.DeepEnsemble)

    def test_create_model(self):
        """
        Test to cover creating a model
        """
        # Create instance of KDESol
        model = KDESol(weigths_path=None)
        # Check that model was created
        assert model.model is not None
        assert isinstance(model.model, tuple)
        # assert isinstance(model.model[0], keras.engine.functional.Functional)
        # assert isinstance(model.model[1], keras.engine.functional.Functional)
        # assert isinstance(model.model[2], keras.engine.functional.Functional)

    # @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Synonyms file is not available in GitHub")
    @pytest.mark.parametrize('input',['CCOOO', 'CNOCNO'])
    def test_predict(self, input):
        """
        Test to cover predicting solubility
        """
        # Create instance of KDESol
        model = KDESol()
        # Predict solubility
        solubility = model(input)
        # Check that solubility was predicted
        assert solubility is not None
        assert isinstance(solubility, np.ndarray)
        assert isinstance(solubility[0], np.float32)
        assert isinstance(solubility[1], np.float32)
        assert isinstance(solubility[2], np.float32)

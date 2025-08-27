import numpy as np
import pandas as pd
import pickle
import pytest

import main
from pse_environments.cstr import CSTR1
from dream_models import MLP


@pytest.fixture
def settings(tmp_path):
    """Provide minimal settings and synthetic test data for pytest."""
    s = main.get_settings()
    # convenience alias used by test scripts
    s['process'] = s['environment']['process']

    # create synthetic test dataset
    state_names = ['c', 'T']
    control_names = ['roh', 'Fc']
    data = pd.DataFrame(
        np.zeros((5, len(state_names) + len(control_names))),
        columns=state_names + control_names,
    )
    test_data = {0: data}
    data_path = tmp_path / 'test_data.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump(test_data, f)
    s['test_data_path'] = str(data_path)

    # populate process-specific metadata expected by tests
    s['CSTR1'] = {
        'state_names': state_names,
        'control_names': control_names,
        'n_states': len(state_names),
        'n_controls': len(control_names),
    }
    # ensure testing options exist
    s.setdefault('dream_model_testing', {})
    s['dream_model_testing']['plot_test'] = False
    s['model_name'] = 'test_model'
    return s


@pytest.fixture
def env(settings):
    """Return a minimal environment wrapper exposing the process model."""
    process_model = CSTR1(delta_t=settings['delta_t'])

    class DummyEnv:
        def __init__(self, process_model):
            self.process_model = process_model

    return DummyEnv(process_model)


@pytest.fixture
def model(env, settings):
    """Instantiate a small neural network model for testing."""
    pm = env.process_model
    model = MLP(
        in_features=pm.n_states + pm.n_actions,
        out_features=pm.n_states,
        hidden_layer_sizes=[8],
        predict_delta=True,
        use_weight_normalization=False,
        device=settings['device'],
    )
    return model

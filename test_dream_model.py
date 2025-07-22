# 3rd party imports
import torch as T
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from common import get_default_device

# NOTE: If you load, deepcopy, or assign a model/ensemble, always call move_model_to_device(model, device) on both the ensemble and each submodel.
# If you ever load or copy a model, do this after the operation.


def test_model(model, env, settings):
    # load data
    with open(settings['test_data_path'], 'rb') as handle:
        test_datasets = pickle.load(handle)

    # get necessary variables
    device = settings['device'] if 'device' in settings else get_default_device()
    state_names = env.process_model.state_names
    control_names = env.process_model.control_names

    # get normalizers
    X_normalizer = env.process_model.state_scaler
    U_normalizer = env.process_model.action_scaler

    # set up MSEs
    MSEs = {k: [] for k in env.process_model.state_names}
    MSEs['total'] = []

    # loop through test datasets
    for i_tds, tds in test_datasets.items():
        # get X and U data from dataset
        X = tds.loc[:, state_names].to_numpy()
        U = tds.loc[:, control_names].to_numpy()

        # normalize data
        X = X_normalizer.scale(X)
        U = U_normalizer.scale(U)

        # ensure numpy arrays for any NumPy ops
        if isinstance(X, T.Tensor):
            X = X.cpu().numpy()
        if isinstance(U, T.Tensor):
            U = U.cpu().numpy()

        # convert to tensors
        X = T.tensor(X, dtype=T.float32, device=device)
        U = T.tensor(U, dtype=T.float32, device=device)

        # simulate entire timeseries using model
        x = X[0, :]
        X_pred = T.zeros_like(X)
        X_pred[0, :] = x
        for t in range(X.shape[0]-1):
            x = model(x, U[t, :])
            X_pred[t+1, :] = x

        # calculate MSEs
        for i, name in enumerate(state_names):
            MSEs[name].append(model.loss_function(X_pred[:, i], X[:, i]).item())
        MSEs['total'].append(model.loss_function(X_pred, X).item())

        # plot results
        if settings['dream_model_testing']['plot_test']:
            # ensure numpy for plotting
            X_plot = X.cpu().numpy() if isinstance(X, T.Tensor) else X
            U_plot = U.cpu().numpy() if isinstance(U, T.Tensor) else U
            X_pred_plot = X_pred.cpu().numpy() if isinstance(X_pred, T.Tensor) else X_pred
            plot_test_trajectory(X_plot, U_plot, X_pred_plot,
                                 {k: v[-1] for k, v in MSEs.items()},
                                 i_tds, settings)

    # print results
    print(f"\nTest results for {settings['model_name']}:")
    for name in settings[settings['process']]['state_names']:
        print(f"{name} MSE: {np.mean(MSEs[name])}")
    print(f"Total MSE: {np.mean(MSEs['total'])}")

    return MSEs

def plot_test_trajectory(X, U, X_pred, MSEs, i_tds, settings):
    # convert to pandas dataframes
    X = pd.DataFrame(X.detach().cpu().numpy(), columns=settings[settings['process']]['state_names'])
    U = pd.DataFrame(U.detach().cpu().numpy(), columns=settings[settings['process']]['control_names'])
    X_pred = pd.DataFrame(X_pred.detach().cpu().numpy(), columns=settings[settings['process']]['state_names'])

    # set up figure
    fig, ax = plt.subplots(X.shape[1] + U.shape[1], 1,
                           sharex=True, sharey=False,
                           figsize=(10,10))
    linewidth = 0.8
    fig.suptitle(f"Overall MSE: {MSEs['total']:.7f}")

    # plot states
    for i, name in enumerate(settings[settings['process']]['state_names']):
        ax[i].plot(X[name], label='true', linewidth=linewidth)
        ax[i].plot(X_pred[name], label='pred', linewidth=linewidth)
        ax[i].set_ylabel(name)
        ax[i].legend()
        ax[i].set_title(f"State {name}, MSE: {MSEs[name]:.7f}")

    # plot controls
    for i, name in enumerate(settings[settings['process']]['control_names']):
        ax[i+X.shape[1]].plot(U[name], label='true', linewidth=linewidth)
        ax[i+X.shape[1]].set_ylabel(name)
        ax[i+X.shape[1]].legend()
        ax[i+X.shape[1]].set_title(f"Control {name}")

    plt.xlabel('Time steps')
    plt.tight_layout()
    plt.savefig(f"{settings['plots_dir']}test_{i_tds}.pdf")
    plt.close(fig)

    return None

if __name__ == "__main__":
    # testing
    model = None
    import main
    settings = main.get_settings()
    device = settings['device'] if 'device' in settings else get_default_device()
    test_model(model, settings, device)
    print('Done!')
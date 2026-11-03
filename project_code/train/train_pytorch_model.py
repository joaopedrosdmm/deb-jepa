import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import inspect

from ..algorithms.loss_functions import MaskedMSELoss, MSEInfeasibilityLoss
from ..algorithms.neural_networks import DEBNet, DEBNetHC
from ..data.load_data import load_dataframes, load_col_types
from ..data.prepare_data_pytorch import prepare_data_tensors
from ..evaluate.prediction_error import evaluate_parameter_predictions_on_data, evaluate_pytorch_model, \
    METRIC_LABEL_TO_NAME
from ..utils.models import save_pytorch_model
from ..utils.results import create_results_directories_for_dataset

from ray import train, tune


def adjust_config(config):
    # Adjust config
    if 'shared_hidden_layers' not in config and 'n_shared_layers' in config and 'shared_hidden_size':
        config['shared_hidden_layers'] = config['n_shared_layers'] * [config['shared_hidden_size']]
    if 'par_hidden_layers' not in config and 'n_par_layers' in config and 'par_hidden_size' in config:
        config['par_hidden_layers'] = config['n_par_layers'] * [config['par_hidden_size']]
    if 'lambdas' not in config:
        config['lambdas'] = [1, 1, 1, 1]
        if 'lambda_kap' in config:
            config['lambdas'][1] = config['lambda_kap']
        if 'lambda_s_H' in config:
            config['lambdas'][2] = config['lambda_s_H']
        if 'lambda_s_p' in config:
            config['lambdas'][3] = config['lambda_s_p']

    return config


def train_neural_network(config, model_class, dataset_name,
                         device=None, seed=42, verbose=False, print_val_metrics_every=10,
                         report_val_scores=False, datasets_folder='data/processed',
                         return_training_stats=False):
    """Optionally return actual step/epoch counts alongside the usual model and metrics."""
    if return_training_stats and report_val_scores:
        raise ValueError('training stats require report_val_scores=False')
    torch.manual_seed(seed=seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    dataframes, col_types = load_dataframes(dataset_name=dataset_name, data_split='train_val_test',
                                            datasets_folder=datasets_folder)
    # col_types = load_col_types(dataset_name=dataset_name, datasets_folder=datasets_folder)
    create_results_directories_for_dataset(dataset_name)

    data_tensors, dataloaders, datasets, scalers = prepare_data_tensors(data=dataframes, col_types=col_types,
                                                                        batch_size=config['batch_size'],
                                                                        scaling_type=config['scaling_type'],
                                                                        device=device)

    # Adjust config
    config = adjust_config(config)

    # Build model
    model_args = dict(col_types=col_types,
                      shared_hidden_layers=config['shared_hidden_layers'],
                      par_hidden_layers=config['par_hidden_layers'],
                      dropout_prob=config['dropout_prob'],
                      output_transformer=scalers['output'],
                      input_transformer=scalers['input'],
                      use_skip_connection=config['use_skip_connection'],
                      )
    if 'clamp_function' in config and 'clamp_function' in inspect.signature(model_class.__init__).parameters:
        model_args['clamp_function'] = config['clamp_function']
    model = model_class(**model_args).to(device)

    # Create loss function
    if config['loss_function'] == 'mse_infeasibility':
        criterion = MSEInfeasibilityLoss(col_types=col_types,
                                         output_transformer=scalers['output'],
                                         lambdas=torch.tensor(config['lambdas'], dtype=torch.float32),
                                         )
    else:
        criterion = MaskedMSELoss()
    criterion.to(device)

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Get training configuration from config
    max_epochs = config['max_epochs']
    early_stopping_metric = config['early_stopping_metric']
    patience = config['patience']
    n_epochs_gradient_descent = config['n_epochs_gradient_descent']
    # Initialize variables
    metric_name_list = list(METRIC_LABEL_TO_NAME.values())
    traces_df = pd.DataFrame(index=range(max_epochs), columns=['train_loss', 'val_loss'] + metric_name_list)
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0

    optimizer_steps = 0
    epochs_completed = 0

    # Train the model with batches
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        model.train()
        epoch_loss = 0
        for i, (X_batch, y_batch, mask) in enumerate(dataloaders['train']):
            # Get model outputs
            outputs = model(X_batch)
            # Compute loss
            loss = criterion(outputs, y_batch, mask)
            loss.backward()

            epoch_loss += loss.item()
            #  Do some epochs of full gradient descent before starting with SGD
            if epoch < n_epochs_gradient_descent:
                if i == len(dataloaders['train']) - 1:
                    optimizer.step()
                    optimizer_steps += 1
                    optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer_steps += 1
                optimizer.zero_grad()
        traces_df.loc[epoch, 'train_loss'] = epoch_loss / len(dataloaders['train'])

        # Evaluate on validation set
        traces_df.loc[epoch, 'val_loss'] = evaluate_pytorch_model(dataloader=dataloaders['val'],
                                                                  model=model,
                                                                  criterion=criterion,
                                                                  col_types=col_types)

        epoch_metrics_df = evaluate_parameter_predictions_on_data(
            data=data_tensors['val'],
            col_types=col_types,
            model=model,
            print_score=verbose and (epoch + 1) % print_val_metrics_every == 0,
        )
        traces_df.loc[epoch, metric_name_list] = epoch_metrics_df.mean(axis=1).rename(METRIC_LABEL_TO_NAME)
        
        if verbose:
            print(f"\nEpoch [{epoch + 1}/{max_epochs}], "
                f"Loss: {traces_df.loc[epoch, 'train_loss']:.4f}, "
                f"Val Loss: {traces_df.loc[epoch, 'val_loss']:.4f}, "
                f"Val logQ: {traces_df.loc[epoch, 'logQ']:.4f}, "
                f"Val sMAPE: {traces_df.loc[epoch, 'sMAPE']:.4f}, "
                f"Val MAPE: {traces_df.loc[epoch, 'MAPE']:.4f}, "
                f"{epochs_no_improve}")
            if config['loss_function'] == 'mse_infeasibility':
                print(criterion.loss_values)

        epochs_completed = epoch + 1
        if report_val_scores:
            report_metrics_dict = traces_df.loc[epoch].to_dict()
            report_metrics_dict['epoch'] = epoch
            report_metrics_dict.pop('train_loss')
            train.report(report_metrics_dict)
        if early_stopping_metric is not None:
            if traces_df.loc[epoch, early_stopping_metric] <= traces_df.loc[best_epoch, early_stopping_metric]:
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping check
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch}. "
                      f"Best {early_stopping_metric}: {traces_df.loc[best_epoch, early_stopping_metric]:.4f}")
                break

    if early_stopping_metric is None:
        report_epoch = max_epochs - 1
    else:
        # End of training - restore best model state if not already set
        model.load_state_dict(best_model_state)
        report_epoch = best_epoch
    if report_val_scores:
        report_metrics_dict = traces_df.loc[report_epoch].to_dict()
        report_metrics_dict.pop('train_loss')
        report_metrics_dict['epoch'] = report_epoch
        return report_metrics_dict
    else:
        if return_training_stats:
            return model, traces_df.loc[report_epoch], {
                "optimizer_steps": optimizer_steps, "epochs_completed": epochs_completed,
                "selected_epoch": report_epoch + 1}
        return model, traces_df.loc[report_epoch]


if __name__ == '__main__':
    dataset_name = 'init_net'


    config = {
        'scaling_type': 'log_standardize',
        'batch_size': 1,
        'max_epochs': 675,
        'learning_rate': 1.1e-4,
        'weight_decay': 3.17e-2,
        'shared_hidden_layers': 0 * [64],
        'par_hidden_layers': 2 * [128],
        'lambdas': [1, 1, 1, 1],
        'loss_function': 'mse',
        'output_weight_strategy': '',
        # 'dropout_prob': 0,
        'dropout_prob': 0.24,
        'clamp_function': 'logsigmoid',
        'use_skip_connection': False,
        'n_epochs_gradient_descent': 0,
        'patience': 10,
        # 'early_stopping_metric': 'logQ',
        'early_stopping_metric': None,
    }
    print(config)

    model_name = 'MLP'
    model_class = DEBNetHC
    save_model = True
    save_folder = f'results/{dataset_name}'
    evaluate_on_test = True

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model, best_epoch_metrics_df = train_neural_network(
        model_class=model_class,
        config=config,
        dataset_name=dataset_name,
        verbose=True,
        report_val_scores=False,
        device=device,
    )
    formatted_metric = 'logQ'
    formatted_score = format(best_epoch_metrics_df[formatted_metric], '.4e').replace('.', '')
    best_model_name = f'{formatted_metric}_{formatted_score}_{model_name}'
    test_performance_save_folder = os.path.join(save_folder, 'test_performance')
    test_performance_save_file = os.path.join(test_performance_save_folder, f"{best_model_name}.csv")

    if save_model:
        save_pytorch_model(model=model, folder=os.path.join(save_folder, 'models'), filename=best_model_name)

    if evaluate_on_test:
        # Load and preprocess data
        dfs, col_types = load_dataframes(dataset_name=dataset_name, data_split='train_val_test')
        data_tensors, dataloaders, datasets, scalers = prepare_data_tensors(data=dfs, col_types=col_types,
                                                                            batch_size=config['batch_size'],
                                                                            scaling_type=config['scaling_type'],
                                                                            device=device,
                                                                            )

        # Evaluate parameter predictions in parameter scale
        model.eval()
        evaluate_parameter_predictions_on_data(
            data=data_tensors['test'],
            col_types=col_types,
            model=model,
            print_score=True,
            save_score=save_model,
            results_save_path=test_performance_save_file,
            return_std=True,
        )

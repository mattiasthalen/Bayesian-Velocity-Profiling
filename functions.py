import time
import pint_xarray
import warnings
import os

import sklearn.preprocessing
import sklearn.metrics

import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import xarray as xr

from collections import OrderedDict
from urllib.parse import unquote

from urllib.parse import unquote
from matplotlib import pyplot as plt
import matplotlib as mpl

from typing import Callable, Optional, Dict, List, Union, NoReturn

### Miscellaneous functions
def extract_kpis(velocity, load):

    mask = load < 0
    mean_mu_masked = np.ma.masked_where(mask, load).compressed()
    velocity_masked = np.ma.masked_where(mask, velocity).compressed()

    zero_load_velocity = velocity_masked.max()
    zero_velocity_load = mean_mu_masked.max()
    area_under_curve = sklearn.metrics.auc(velocity_masked, mean_mu_masked)

    kpis = np.array([zero_load_velocity, zero_velocity_load, area_under_curve]).round(2)

    return kpis

def create_kpi_dataset(inference_data, df):

    sessions = df['session'].unique()
    zero_load_velocity = []
    zero_velocity_load = []
    area_under_curve = []

    mean_predictions = inference_data['predictions']['mu_std_rescaled'].mean(['chain', 'draw'])
    x_observations = inference_data['predictions_constant_data']['velocity_std_rescaled']

    for session in sessions:
        filter = df['session'] == session
        observations = df[filter]['observation'].values

        y_model_mean = mean_predictions.sel(observation = observations)
        x = x_observations.sel(observation = observations)

        kpis = extract_kpis(x, y_model_mean)
        zero_load_velocity.append(kpis[0])
        zero_velocity_load.append(kpis[1])
        area_under_curve.append(kpis[2])
    
    dict = {
        'coords': {
            'session': {
                'dims': 'session',
                'data': sessions,
            }
        },
        'dims': 'session',
        'data_vars': {
            'zero_load_velocity': {
                'dims': 'session',
                'data': zero_load_velocity
            },
            'zero_velocity_load': {
                'dims': 'session',
                'data': zero_velocity_load
            },
            'area_under_curve': {
                'dims': 'session',
                'data': area_under_curve
            }
        }
    }
    
    dataset = xr.Dataset.from_dict(dict)

    return dataset

def assign_set_type(da, **kwargs):
    set_category = xr.where(da['set'] < da.idxmax('set'), 'Work Up', np.nan)
    set_category = xr.where(da['set'] == da.idxmax('set'), 'Top Set', set_category)
    set_category = xr.where(da['set'] > da.idxmax('set'), 'Back Off', set_category)
    set_category = xr.where(np.isnan(da), np.nan, set_category)
    
    return set_category

def agg_summarize(x, **kwargs):
    x = x[np.isfinite(x)]

    if len(x) > 0:
        min = np.min(x)
        max = np.max(x)
        first = x[0]
        last = x[-1]
        peak_end = np.mean([min, last])
        mean = np.mean(x)
        median = np.median(x)
        hdi = az.hdi(x)
        result = np.array([min, max, first, last, peak_end, mean, median, *hdi])
    else:
        result = np.array([np.nan]*9)
    
    return result

def summarize(x, reduce_dim = 'rep', **kwargs):
    summaries = xr.apply_ufunc(agg_summarize,
                            x,
                            vectorize = True,
                            input_core_dims = [[reduce_dim]],
                            output_core_dims = [['aggregation']])
    
    summaries['aggregation'] = ['min', 'max', 'first', 'last', 'peak_end', 'mean', 'median', 'hdi_lower', 'hdi_upper']
    
    return summaries

def agg_hdi_summary(x, **kwargs):
    mean = x.mean()
    median = np.median(x)
    hdi = az.hdi(x)

    return np.array([mean, median, *hdi])

def hdi_summary(x, reduce_dim = 'sample', **kwargs):
    try:
        x = x.pint.dequantify()
    except:
        pass

    summaries = xr.apply_ufunc(agg_hdi_summary,
                            x,
                            vectorize = True,
                            input_core_dims = [[reduce_dim]],
                            output_core_dims = [['hdi_aggregation']])
    
    summaries['hdi_aggregation'] = ['mean', 'median', 'hdi_lower', 'hdi_upper']
    
    return summaries

def all_nan_summary(x, mode = 'mean', **kwargs):
    if np.all(np.isnan(x)):
        return np.nan
    elif mode == 'max':
        return np.nanmax(x)
    elif mode == 'min':
        return np.nanmin(x)
    elif mode == 'mean':
        return np.nanmean(x)
    elif mode == 'sum':
        return np.nansum(x)
        
def all_nan_max(x, **kwargs):
    return all_nan_summary(x, 'max', **kwargs)
        
def all_nan_min(x, **kwargs):
    return all_nan_summary(x, 'min', **kwargs)
        
def all_nan_mean(x, **kwargs):
    return all_nan_summary(x, 'mean', **kwargs)
        
def all_nan_sum(x, **kwargs):
    return all_nan_summary(x, 'sum', **kwargs)

### ETL functions
def csv_to_dataframe(csv_path, **kwargs):
    df = pd.read_csv(csv_path)

    df.columns = (df.columns
                    .str.lower()
                    .str.replace(' \(m/s\)', '', regex = True)
                    .str.replace(' \(mm\)', '', regex = True)
                    .str.replace(' \(sec\)', '', regex = True)
                    .str.replace(' \(%\)', '', regex = True)
                    .str.replace(' ', '_', regex = True))
        
    df.rename(columns = {'weight': 'load'}, inplace = True)

    df['workout_start_time'] = pd.to_datetime(df['workout_start_time'], format = '%d/%m/%Y, %H:%M:%S')

    df.dropna(subset = ['exercise'], inplace = True)
    df['rest_time'] = pd.to_timedelta(df['rest_time'])

    # Correct split session
    df['set'].mask((df['exercise'] == 'deadlift') & (df['workout_start_time'] == pd.to_datetime('2020-12-30 13:06:04')), df['set'] + 7, inplace = True)
    df.replace({'workout_start_time': pd.to_datetime('2020-12-30 13:06:04')}, pd.to_datetime('2020-12-30 12:53:09'), inplace = True)
    df.replace({'workout_start_time': pd.to_datetime('2021-01-07 11:50:22')}, pd.to_datetime('2021-01-07 11:20:07'), inplace = True)
    df.replace({'workout_start_time': pd.to_datetime('2021-06-10 12:02:22')}, pd.to_datetime('2021-06-10 11:56:31'), inplace = True)
    df.replace({'workout_start_time': pd.to_datetime('2021-06-14 12:06:00')}, pd.to_datetime('2021-06-14 11:57:50'), inplace = True)

    # Reindex sets & reps to counter bugs in the extract
    df['set'] = df.groupby(['exercise', 'workout_start_time'], group_keys = False)['set'].apply(lambda x: (x != x.shift()).cumsum() - 1)
    df['rep'] = df.groupby(['exercise', 'workout_start_time', 'set'], group_keys = False).cumcount()

    # Convert from , to . as decimal sign
    df['load'] = df['load'].str.replace(',', '.').astype('float')

    # Drop rows with tag fail
    fail_filter = df['tags'].str.contains('fail', na = False)
    df = df[~fail_filter]

    # Handle the case when a rep is split into two reps
    rep_split_filter = df['tags'].str.contains('rep split', na = False)

    rep_split_group_agg = ['exercise', 'workout_start_time', 'set', 'load', 'metric', 'rest_time', 'tags']

    rep_split_df = (df[rep_split_filter].groupby(rep_split_group_agg, group_keys = False)
                                        .aggregate({'range_of_motion': 'sum',
                                                    'duration_of_rep': 'sum',
                                                    'peak_velocity': 'max'}))
    rep_split_df['avg_velocity'] = rep_split_df['range_of_motion']/1000/rep_split_df['duration_of_rep']
    rep_split_df['rep'] = 0
    rep_split_df.reset_index(inplace = True)

    rep_split_group = ['exercise', 'workout_start_time', 'set', 'rep']

    rep_split_df = (rep_split_df.groupby(rep_split_group, group_keys = False)
                                .max()
                                .reset_index())

    df = pd.concat([df[~rep_split_filter], rep_split_df])

    df['avg_velocity'] = df['avg_velocity'].round(2)

    # Group to get multi index
    df = df.groupby(rep_split_group, group_keys = False).max()

    return df

def dataframe_to_dataset(df, **kwargs):
    # Convert to xarray
    ds = df.to_xarray()

    # Change Set and Rep to integers
    ds['set'] = ds['set'].astype(int)
    ds['rep'] = ds['rep'].astype(int)

    # Move variables to coords
    ds = ds.set_coords(['metric', 'tags'])

    # Define UOMs
    ds = ds.pint.quantify({'load': 'kg',
                            'avg_velocity': 'meter / second',
                            'peak_velocity': 'meter / seconds',
                            'range_of_motion': 'mm',
                            'duration_of_rep': 's'})

    # Session meta data
    session_stack = ['exercise', 'workout_start_time']
    ds['session_max_load'] = ds['load'].stack(stack = session_stack)\
                                        .groupby('stack')\
                                        .reduce(all_nan_max, ...)\
                                        .unstack()

    # Set meta data
    set_stack = [*session_stack, 'set']
    ds['load'] = (ds['load'].stack(stack = set_stack)
                            .groupby('stack')
                            .reduce(all_nan_max, ...)
                            .unstack())

    ds['reps'] = (ds['avg_velocity'].stack(stack = set_stack)
                                    .groupby('stack')
                                    .count(...)
                                    .unstack()
                                    .where(ds['load'] > 0, drop = True))

    ds['set_velocities'] = summarize(ds['avg_velocity'].pint.dequantify())
    ds['set_velocities'] = ds['set_velocities'].pint.quantify({ds['set_velocities'].name: 'mps'})

    ds.coords['set_type'] = assign_set_type(ds['load'])

    # Add the running min top set velocity per exercise
    ds['minimum_velocity_threshold'] = (ds['set_velocities'].sel({'aggregation': 'first'})
                                                            .where(ds.coords['set_type'] == 'Top Set')
                                                            .pint.dequantify()
                                                            .stack(stack = session_stack)
                                                            .groupby('stack')
                                                            .reduce(all_nan_min, ...)
                                                            .unstack()
                                                            .rolling({'workout_start_time': len(ds['workout_start_time'])},
                                                                    min_periods = 1)
                                                            .min())
    ds['minimum_velocity_threshold'] = ds['minimum_velocity_threshold'].pint.quantify({ds['minimum_velocity_threshold'].name: 'meter / second'})

    # Add running max load per exercise
    ds['rolling_max_load'] = (ds['load'].pint.dequantify()
                                        .stack(stack = session_stack)
                                        .groupby('stack')
                                        .reduce(all_nan_max, ...)
                                        .unstack()
                                        .rolling({'workout_start_time': len(ds['workout_start_time'])},
                                                min_periods = 1)
                                        .max())
    ds['rolling_max_load'] = ds['rolling_max_load'].pint.quantify({ds['rolling_max_load'].name: 'kg'})

    # Additional session meta data
    ds['workup_sets'] = ds['load'].where(ds.coords['set_type'] == 'Work Up', drop = True)\
                                    .stack(stack = session_stack)\
                                    .groupby('stack')\
                                    .count(...)\
                                    .unstack()

    #ds['session_regression_coefficients'] = linear_fit(ds, 'load', 'set_velocities', 'set')

    #ds['estimated_1rm'] = linear_predict(ds['minimum_velocity_threshold'].pint.dequantify(), ds['session_regression_coefficients'])
    #ds['estimated_1rm'] = ds['estimated_1rm'].pint.quantify({ds['estimated_1rm'].name: 'kg'})

    #ds['zero_velocity_load'] = linear_predict(0, ds['session_regression_coefficients'])
    #ds['zero_velocity_load'] = ds['zero_velocity_load'].pint.quantify({ds['zero_velocity_load'].name: 'kg'})

    #ds['zero_load_velocity'] = linear_predict(0, ds['session_regression_coefficients'], reverse = True)
    #ds['zero_load_velocity'] = ds['zero_load_velocity'].pint.quantify({ds['zero_load_velocity'].name: 'mps'})

    #ds['curve_score'] = ds['zero_velocity_load'].pint.dequantify()*ds['zero_load_velocity'].pint.dequantify()/2

    ds['session_volume'] = (ds['load'] * ds['reps']).stack(stack = session_stack).groupby('stack').sum(...).unstack()
    #ds['session_relative_volume'] = ds['session_volume']/ds['estimated_1rm']

    # Rep meta data
    #ds['rep_exertion'] = linear_predict(ds['avg_velocity'].pint.dequantify(), ds['session_regression_coefficients'])/ds['estimated_1rm'].pint.dequantify()
    #ds['rep_force'] = (ds['load']*ds['range_of_motion'].pint.to('meter')/ds['duration_of_rep']**2).pint.to('N')
    #ds['rep_energy'] = (ds['rep_force']*ds['range_of_motion'].pint.to('meter')).pint.to('J')

    # Session meta data
    #ds['session_exertion_load'] = ds['rep_exertion'].stack(stack = ['exercise', 'workout_start_time']).groupby('stack').reduce(all_nan_sum, ...).unstack().pint.dequantify()

    # Add PR coordinates
    ds.coords['max_load_pr_flag'] = ds['rolling_max_load'].diff('workout_start_time') > 0
    ds.coords['max_load_pr_flag'] = ds.coords['max_load_pr_flag'].fillna(0).astype(int)

    # Add indexing for inference
    session_shape = [ds.dims[i] for i in session_stack]
    ds.coords['session'] = (session_stack, np.arange(np.prod(session_shape)).reshape(session_shape))

    observation_shape = [ds.dims[i] for i in set_stack]
    ds.coords['observation'] = (set_stack, np.arange(np.prod(observation_shape)).reshape(observation_shape))

    return ds

def load_data(csv_path, **kwargs):
    df = csv_to_dataframe(csv_path, **kwargs)
    ds = dataframe_to_dataset(df, **kwargs)

    return ds

def merge_by_coord_dimension(ds_a, ds_b, coord_dimension, data_vars):
    ds_merged = ds_a.copy()

    shape = ds_merged[coord_dimension].shape
    dims = ds_merged[coord_dimension].dims

    for data_var in data_vars:
        data = np.full_like(ds_merged[coord_dimension].values.flatten().astype('float64'), np.nan)

        for coord in ds_merged[coord_dimension].values.flatten():
            if coord in ds_b[coord_dimension].values.flatten():
                data[coord] = ds_b.sel({coord_dimension: coord})[data_var].values.flatten()

        ds_merged[data_var] = (dims, data.reshape(shape))

    return ds_merged

def extract_training_data(ds, **kwargs):
    df = (ds[['load', 'set_velocities', 'observation', 'session']]
                .pint.dequantify()
                .sel({'aggregation': 'max'}, drop = True)
                .where(ds.coords['set_type'] != 'Back Off')
                .where(ds.coords['exercise'] != 'front squat')
                .drop_vars(['set_type', 'max_load_pr_flag'])
                .to_dataframe()
                .dropna()
                .reset_index()
                .drop(['set', 'workout_start_time'], axis = 1)
                .rename(columns = {'set_velocities': 'velocity'}))

    # Scale data to simplify inference
    velocity_standardizer = sklearn.preprocessing.StandardScaler()
    df['velocity_std'] = velocity_standardizer.fit_transform(df['velocity'].values.reshape(-1, 1))

    load_standardizer = sklearn.preprocessing.StandardScaler()
    df['load_std'] = load_standardizer.fit_transform(df['load'].values.reshape(-1, 1))


    return [df, velocity_standardizer, load_standardizer]

def extract_model_input(df):
    velocity_std = df['velocity_std'].values
    load_std = df['load_std'].values

    observation = df['observation'].values

    exercises = df['exercise'].values
    exercise_encoder = sklearn.preprocessing.LabelEncoder()
    exercise_encoder.fit(exercises)

    sessions = df['session'].values
    session_encoder = sklearn.preprocessing.LabelEncoder()
    observation_session_idx = session_encoder.fit_transform(sessions)

    session_exercise = (df.reset_index()[['session', 'exercise']]
                        .drop_duplicates()
                        .set_index('session', verify_integrity = True)
                        .sort_index()['exercise']
                        .values)

    session_exercise_idx = exercise_encoder.transform(session_exercise)

    # Needs to be in order from observation and up to global
    coords = {'observation': observation,
            'session': session_encoder.classes_,
            'exercise': exercise_encoder.classes_}

    model_data = {'velocity_std': velocity_std,
                'load_std': load_std,
                'observation_session_idx': observation_session_idx,
                'session_exercise_idx': session_exercise_idx,
                'coords': coords}
    
    return [model_data, session_encoder, exercise_encoder]

def generate_prediction_data(df, velocity_standardizer):
    df = df.copy()

    velocity_min = 0
    velocity_max = np.ceil(df['velocity'].max())
    velocity_round = 0.01
    velocity_pred = np.linspace(velocity_max, velocity_min, int(velocity_max/velocity_round) + 1).round(2)
    
    prediction_df = pd.DataFrame()

    for i in df['session'].unique():
        session = np.full_like(velocity_pred, i).astype(int)
        new_data = pd.DataFrame({'velocity': velocity_pred,
                                'session': session})
        
        prediction_df = pd.concat([prediction_df, new_data])

    prediction_df['velocity_std'] = velocity_standardizer.transform(prediction_df['velocity'].values.reshape(-1, 1))

    prediction_df = prediction_df.merge(df[['velocity', 'session', 'load_std', 'observation']], on = ['velocity', 'session'], how = 'left')

    last_observation = prediction_df['observation'].max()
    observation = []

    for i in prediction_df['observation']:

        obs = i

        if i >= 0:
            obs = i
        else:
            obs = last_observation + 1
            last_observation = obs
        
        observation.append(obs)

    prediction_df['observation'] = np.array(observation).astype(int)

    prediction_df = prediction_df.merge(df.groupby(['session'])['exercise'].max(), on = 'session')

    return prediction_df

def rescale_inference_data(inference_data, velocity_scaler, load_scaler):
    inference_data = inference_data.copy()

    groups = {'predictions': [['mu_std', load_scaler],
                            ['likelihood_std', load_scaler]],
            'predictions_constant_data': [['velocity_std', velocity_scaler],
                                            ['load_std', load_scaler]]}

    for group, vars in groups.items():

        if group not in inference_data.groups():
            print(f'The group {group}) was not found in the inference data.')
        
        else:
            ds = getattr(inference_data, group)

            for var in vars:
                var_name, var_scaler = var

                if var_name not in list(ds.data_vars):
                    print(f'The data var {var_name} was not found in the group {group}.')
                
                else:
                    array = ds[var_name]
                    dims = array.dims
                    values = array.values
                    shape = values.shape

                    rescaled_values = var_scaler.inverse_transform(values.reshape(-1, 1)).reshape(shape)

                    ds[f'{var_name}_rescaled'] = (dims, rescaled_values)
                
    return inference_data

### Model functions
def sample_pymc_nuts(model,
                    prior_predictive = True,
                    posterior_predictive = True,
                    **kwargs):
    with model:
        print('Sample posterior...')
        inference_data = pm.sample(**kwargs)
        
        if prior_predictive:
            print('Sample prior predictive...')
            inference_data.extend(pm.sample_prior_predictive())

        if posterior_predictive:
            print('Sample posterior predictive...')
            inference_data.extend(pm.sample_posterior_predictive(trace = inference_data))

    return inference_data

def sample_model(model, csv_path, inference_path):
    try:
        csv_mtime = os.path.getmtime(csv_path)
        nc_mtime = os.path.getmtime(inference_path)

        if csv_mtime > nc_mtime:
            raise Exception('New data available.')

        inference_data = az.from_netcdf(inference_path)
        print('Inference data loaded.')

    except (FileNotFoundError, Exception) as e:
        print(e)
        inference_data = sample_pymc_nuts(model = model, target_accept = 0.9)
        inference_data.to_netcdf(inference_path)
    
    return inference_data

### Plotting functions
def plot_last_session_predictions(inference_data,
                                  df,
                                  exercises,
                                  hdi_prob = 0.8):
    n_exercises = len(exercises)

    n_rows = np.ceil(n_exercises / 2).astype(int)
    n_cols = n_exercises - n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize = (15, 10), sharex = True, sharey = True)

    axes = axes.flatten()

    for i, ax in enumerate(axes):

        exercise = exercises[i]

        last_session = df[df['exercise'] == exercise]['session'].max()
        last_session_observations = df[df['session'] == last_session]['observation'].values

        predictions = inference_data.predictions.sel(observation = last_session_observations)
        predictions_constant_data = inference_data.predictions_constant_data.sel(observation = last_session_observations)
        
        x = predictions_constant_data['velocity_std_rescaled']
        y = predictions_constant_data['load_std_rescaled']
        y_hat = predictions['likelihood_std_rescaled']
        y_model = predictions['mu_std_rescaled']
        y_model_mean = y_model.mean(['chain', 'draw'])

        zero_load_velocity, zero_velocity_load, area_under_curve = extract_kpis(x, y_model_mean)

        az.plot_lm(x = x,
                y = y,
                y_hat = y_hat,
                y_model = y_model,
                kind_pp = 'hdi',
                kind_model = 'hdi',
                y_kwargs = {'marker': '.',
                            'markersize': 9,
                            'color': '#0571b0',
                            'label': 'Observed Data',
                            'zorder': 3},
                y_hat_fill_kwargs = {'fill_kwargs': {'zorder': 0,
                                                        'alpha': 0.6},
                                        'color': '#92c5de',
                                        'hdi_prob': hdi_prob},
                y_model_mean_kwargs = {'zorder': 2,
                                        'color': '#ca0020',
                                        'linewidth': 1,
                                        'linestyle': 'dashed'},
                y_model_fill_kwargs = {'zorder': 1,
                                        'color': '#f4a582',
                                        'alpha': 0.6,
                                        #'fill_kwargs': {'zorder': 1},
                                        #'hdi_kwargs': {'hdi_prob': hdi_prob}
                                        },
                legend = False,
                axes = ax)
                
        ax.set_title(exercise.title(), fontweight = 'bold')
        ax.set_xlabel('Velocity', fontweight = 'bold')
        ax.set_ylabel('Load', fontweight = 'bold')

        ax.annotate(text = f'Zero Load Velocity: {zero_load_velocity:.2f}\nZero Velocity Load: {zero_velocity_load:.0f}\nArea Under Curve: {area_under_curve:.0f}',
                    xy = (0.98, 0.7),
                    xycoords = 'axes fraction',
                    horizontalalignment = 'right',
                    verticalalignment = 'top',
                    bbox = {'boxstyle': 'round',
                            'edgecolor': plt.rcParams['axes.edgecolor'],
                            'facecolor': plt.rcParams['axes.facecolor'],
                            'alpha': plt.rcParams['legend.framealpha']}
                    )

        ax.set_xlim(0)
        ax.set_ylim(0)
        
        ax.legend(loc = 'upper right')

    fig.suptitle('LAST SESSION PER EXERCISE',
                fontweight = 'bold',
                fontsize = 'x-large')

    plt.draw()

def plot_pbc(ds, exercise, data_var, window = 20, signal_window = 8, ax = None, display_df = False, **kwargs):
    df = (ds[data_var].sel({'exercise': exercise}, drop = True)
                      .pint.dequantify()
                      .to_dataframe()
                      .dropna())
    
    df['moving_average'] = (df[data_var].sort_index(ascending = False)
                                        .rolling(window, min_periods = 1)
                                        .mean())
    
    df['moving_range'] = (df[data_var].diff(-1)
                                      .abs()
                                      .sort_index(ascending = False)
                                      .rolling(window, min_periods = 1)
                                      .mean())

    df['process_average'] = df['moving_average']
    df['process_range'] = df['moving_range']
    df['signal'] = None
    df['signal_min'] = None
    df['signal_max'] = None
    df['signal_above_average'] = None
    df['signal_below_average'] = None

    n_rows = len(df)
    previous_signal_id = 0

    for row in np.arange(n_rows):
        first_row = row == 0
        sufficient_rows_left = n_rows - row >= window

        signal_start_id = np.max([8, row - signal_window])

        df['signal_min'].iat[row] = df[data_var][signal_start_id:row].min()
        df['signal_max'].iat[row] = df[data_var][signal_start_id:row].max()

        df['signal_above_average'].iat[row] = (df['signal_min'][row] > df['process_average'][row - 1])
        df['signal_below_average'].iat[row] = (df['signal_max'][row] < df['process_average'][row - 1])

        signal_open = (first_row) | (row >= previous_signal_id + window)
        signal = (signal_open) & (sufficient_rows_left) & (first_row | df['signal_above_average'][row] | df['signal_below_average'][row])
        df['signal'].iat[row] = signal
        
        df['process_average'].iat[row] =  df['process_average'][row - 1]
        df['process_range'].iat[row] =  df['process_range'][row - 1]

        if signal:
            previous_signal_id = row
            df['process_average'].iat[row] =  df['moving_average'][row]
            df['process_range'].iat[row] =  df['moving_range'][row]
        else:
            df['process_average'].iat[row] =  df['process_average'][row - 1]
            df['process_range'].iat[row] =  df['process_range'][row - 1]

    zones = 3

    for i in np.arange(zones):
        offset = df['process_range']*(i + 1)/1.128
        df[f'lower_limit_{i}'] = df['process_average'] - offset
        df[f'upper_limit_{i}'] = df['process_average'] + offset

    if display_df:
        display(df)

    if ax is None:
        ax = plt.gca()

    ax.scatter(df.index, df[data_var], marker = '.', alpha = 0.6, color = 'slategray', zorder = 3)
    ax.plot(df.index, df['process_average'], linestyle = '--', color = 'slategray', zorder = 3, linewidth = 1.5)

    colors = {'lower': ['#f7f7f7', '#f4a582', '#ca0020'],
              'upper': ['#f7f7f7', '#92c5de', '#0571b0']}

    for i in np.arange(zones):
        prev_i = np.max([0, i - 1])

        for level in ['lower', 'upper']:
            ax.plot(df.index, df[f'{level}_limit_{i}'], linewidth = 0.5, alpha = 0.9, color = colors[level][i], zorder = 2)

            if i > 0:
                ax.fill_between(df.index,df[f'{level}_limit_{i}'], df[f'{level}_limit_{prev_i}'], alpha = 0.5, color = colors[level][i], zorder = 1, label = f'Zone {i + 1} ({level})')

        if i == 0:
            ax.fill_between(df.index,df[f'lower_limit_{i}'], df[f'upper_limit_{prev_i}'], alpha = 0.5, color = colors['lower'][i], zorder = 1, label = 'Zone 1')

    return ax

def plot_kpis(ds, exercise, vars, figsize = None, n_cols = None, ylim = None, legend = False, **kwargs):
    title_format = lambda x: x.title().replace('_', ' ')

    var_titles = [title_format(var) for var in vars]

    n_vars = len(vars)

    if n_cols is None:
        n_cols = 2 if n_vars > 1 else 1

    n_rows = np.ceil(n_vars/n_cols).astype(int) if n_vars > 2 else 1
    
    figsize = np.array([6, 3]) * [n_cols, n_rows] if figsize is None else figsize

    fig, axes = plt.subplots(ncols = n_cols,
                             nrows = n_rows,
                             constrained_layout = True,
                             figsize = figsize,
                             sharex = True)
    
    axes = axes.flatten() if n_vars > 1 else [axes]

    for key, val in enumerate(vars):
        plot_pbc(ds, exercise, vars[key], ax = axes[key], **kwargs)
        title = var_titles[key]
        axes[key].set_title(title, fontweight = 'bold')
        axes[key].tick_params(labelrotation = 90)
        axes[key].grid(True)
        
        if legend:
            axes[key].legend()

        if ylim is not None:
            axes[key].set_ylim(ylim)

    fig.suptitle(exercise.upper(), fontsize = 16, fontweight = 'bold')
    fig.supxlabel('Workout Start Time')

    plt.draw()
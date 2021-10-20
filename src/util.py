from pymongo import MongoClient as mc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error
import pandas as pd
import numpy as np
from datetime import datetime as dt

from tensorflow.python.ops.gen_math_ops import sqrt
from src.data_process_supporter import *


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]]
                for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


WindowGenerator.split_window = split_window


def plot(self, model=None, plot_col='kw (15min)', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_subplots, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend().remove()

    plt.xlabel('Time [h]')


WindowGenerator.plot = plot


def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds


WindowGenerator.make_dataset = make_dataset


@property
def train(self):
    return self.make_dataset(self.train_df)


@property
def val(self):
    return self.make_dataset(self.val_df)


@property
def test(self):
    return self.make_dataset(self.test_df)


@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.test))
        # And cache it for next time
        self._example = result
    return result


WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

MAX_EPOCHS = 100

# loss function
# tf.keras.losses.MeanSquaredError()
# tf.keras.losses.MeanAbsoluteError()
# tf.keras.losses.Huber()


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))


def compile_and_fit(model, window, EPOCHS=20, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[
                      tf.metrics.MeanAbsoluteError(),
    ])

    history = model.fit(window.train, epochs=EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


class KETI_DB:
    def __init__(self):
        self.mongo_uri = "mongodb://localhost:27017"

        print("connect KETIDB,,,")

        self.client = mc(self.mongo_uri)
        self.keti_db = self.client.keti_pattern_recognition

        self.jungang_col = self.keti_db.jungang_pattern
        self.cluster_col = self.keti_db.cluster_info
        self.weather_col = self.keti_db.weather_info

        print("connect success")

    def __del__(self):
        print("disconnect KETIDB,,,")
        self.client.close()
        print("disconnect success!!")

    def get_jungang_table(self):
        jungang_db_cur = self.jungang_col.find()
        db_datas = [_ for _ in jungang_db_cur]

        jg_datas = pd.DataFrame(columns=['Date Time', 'energy (kw 15min)'])
        jg_datas['Date Time'] = [_['ttime'] for _ in db_datas]
        jg_datas['energy (kw 15min)'] = [_['energy'] for _ in db_datas]

        date_time = pd.to_datetime(jg_datas.pop('Date Time'),
                                   format="%Y-%m-%d %H:%M:%S")
        jg_datas.index = date_time

        # 불 필요 데이터 잘라내기 ( Split Datas )
        idx = jg_datas.index.get_loc(
            jg_datas[jg_datas['energy (kw 15min)'] == 0].index[7])

        jg_datas = jg_datas.iloc[:idx].copy()

        # ~ 2018 year data parsing and basic columns config
        jg_datas = jg_datas[jg_datas.index.year <= 2018]
        date_time = jg_datas.index
        timestamp = date_time.map(dt.timestamp)

        day = 24 * 60 * 60
        week = 7 * day
        year = (365) * day

        jg_datas['week sin'] = calc_sin(timestamp, week)
        jg_datas['week cos'] = calc_cos(timestamp, week)
        jg_datas['year sin'] = calc_sin(timestamp, year)
        jg_datas['year cos'] = calc_cos(timestamp, year)
        jg_datas['season'] = [get_season(_.month) for _ in jg_datas.index]
        jg_datas['season idx'] = [get_season_to_idx(
            _) for _ in jg_datas['season'].values]

        jg_datas = jg_datas[::4]

        weather_df = pd.DataFrame(columns=['weather', 'avg ta', 'avg rhm'])
        for weather in self.weather_col.find():
            try:
                iscs = weather['weather']
            except:
                iscs = "특이사항없음"
            if iscs != "특이사항없음":
                iscs_idx = 0
                for hours in range(0, 24):
                    if (iscs[iscs_idx]['end time'] == ""):
                        iscs[iscs_idx]['end time'] = weather['date'].replace(
                            hour=23)
                    if (hours >= iscs[iscs_idx]['start time'].hour) and \
                            (hours <= iscs[iscs_idx]['end time'].hour):
                        weather_df = weather_df.append({
                            "weather": iscs[iscs_idx]['weather'],
                            "avg ta": weather['avgTa'],
                            "avg rhm": weather['avgRhm']
                        }, ignore_index=True)
                    else:
                        weather_df = weather_df.append({
                            "weather": "특이사항없음",
                            "avg ta": weather['avgTa'],
                            "avg rhm": weather['avgRhm']
                        }, ignore_index=True)

                    if ((iscs_idx + 1) < len(iscs)) and \
                            (iscs[iscs_idx]['end time'].hour < (hours + 1)):
                        iscs_idx += 1
            else:
                for hours in range(0, 24):
                    weather_df = weather_df.append({
                        "weather": "특이사항없음",
                        "avg ta": weather['avgTa'],
                        "avg rhm": weather['avgRhm']
                    }, ignore_index=True)

        all_data_length = len(jg_datas)
        jg_datas['weather'] = list(weather_df['weather'].values)[
            :all_data_length]
        jg_datas['weather idx'] = list(get_weather_to_idx(
            _) for _ in weather_df['weather'].values)[:all_data_length]
        jg_datas['avg ta'] = list(map(float, weather_df['avg ta'].values))[
            :all_data_length]
        jg_datas['avg rhm'] = list(map(float, weather_df['avg rhm'].values))[
            :all_data_length]

        return jg_datas


class CLUSTER_MATCHING:
    def __init__(self, datas):
        db = KETI_DB()

        # Clusterinbg 가져오기 작업
        cur_cluster_result = db.cluster_col.find({
            "uid": "jungang_pattern"
        })
        cluster_result = dict()

        for data in cur_cluster_result:
            in_dict = pd.DataFrame(columns=['Label', 'Weekday'])
            in_dict.index.name = "Date Time"

            season = data['season']
            infos = data['info']

            dtime = [dt.strptime(_['date'], "%Y-%m-%d") for _ in infos]
            labels = [_['label'] for _ in infos]

            for idx, _ in enumerate(dtime):
                label = labels[idx]
                in_dict.loc[_] = [label, _.weekday()]

            cluster_result[season] = in_dict

        self.cluster_pattern_dict = dict()

        for season in SEASONS:
            result = cluster_result[season]
            in_dict = pd.DataFrame(columns=[_ for _ in range(0, 24)])
            in_dict.index.name = "Label"

            labels = list(set(result['Label']))
            for label in labels:
                cluster_pattern = np.array([])
                date_in_labels = result[result['Label'] == label].index
                for date in date_in_labels:
                    idx = datas.index.get_loc(date)
                    pattern = datas.iloc[idx: idx +
                                         24]['energy (kw 15min)'].values
                    cluster_pattern = np.append(cluster_pattern, pattern)
                cluster_pattern = cluster_pattern.reshape(-1, 24).mean(axis=0)
                in_dict.loc[label] = cluster_pattern

            self.cluster_pattern_dict[season] = in_dict

        # Clustering Matching System Config
        self.cluster_dist_dict = dict()
        for season in cluster_result.keys():
            cluster_season_dict = dict()
            week_list = set(cluster_result[season]['Weekday'])
            for week in week_list:
                week_dist = cluster_result[season][
                    cluster_result[season]['Weekday'] == week
                ]['Weekday'].groupby(cluster_result[season]['Label']).count().sort_values(ascending=False)
                week_top_label = week_dist.index[0]

                cluster_season_dict[week] = week_top_label
            self.cluster_dist_dict[season] = cluster_season_dict

        del db

    def matching(self, date, length=24):
        season = get_season(date.month)
        weekday = date.weekday()
        label = self.cluster_dist_dict[season][weekday]
        c_pattern = self.cluster_pattern_dict[season].loc[label].values[:length]

        return c_pattern

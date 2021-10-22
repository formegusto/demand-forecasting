from src.util import *
from src.cluster_utils import WindowGenerator as CWindowGenerator
from src.data_process_supporter import *
from functools import reduce
import IPython
import IPython.display
import tensorflow as tf

DEFAULT_COLUMNS = [
    'energy (kw 15min)',
    'week sin',
    'week cos',
    'year sin',
    'year cos',
    'season idx',
    'avg ta',
    'avg rhm',
    'weather idx'
]
STATISTIC_COLUMNS = [
    'mean distance',
    'mean sim',
    'tss',
    'wss',
    'ecv',
    'mse',
    'mse',
]

EMPTY_COL_EXCEPTION = Exception("Empty arrays are not allowed")
INVALID_COL_EXCEPTION = Exception("Only Required ["
                                  "'energy (kw 15min) ***'\n",
                                  "'week sin'\n",
                                  "'week cos'\n",
                                  "'year sin'\n",
                                  "'year cos'\n",
                                  "'season idx'\n",
                                  "'avg ta'\n",
                                  "'avg rhm'\n")
ESSENTIAL_ENERGY_COL_EXCEPTION = Exception(
    "The \"energy (kw 15min)\" column must be included.")
DUPLICATED_COLUMNS = Exception("Duplicate columns are not allowed.")
SET_WINDOW_PLEASE = Exception("set_window func execute please.")
SET_MODEL_PLEASE = Exception("set_model func execute please.")
PREDICT_COLUMNS = ['energy (kw 15min)']
SET_PREDICT_PLEASE = Exception("set predicts func execute please.")
SEASON_IDX_PLEASE = Exception(
    "\"Season Model\" must have \"season idx column\".")


def col_check(columns):
    print("###### [Notice] columns validation start ###### \n")

    if len(columns) == 0:
        raise EMPTY_COL_EXCEPTION
    if len(set(columns)) != len(columns):
        raise DUPLICATED_COLUMNS
    for col in columns:
        if col not in DEFAULT_COLUMNS:
            raise INVALID_COL_EXCEPTION

    energy_col_idx = columns.index("energy (kw 15min)")
    if energy_col_idx == -1:
        raise ESSENTIAL_ENERGY_COL_EXCEPTION

    print("###### [Notice] columns validation success ###### \n")

    return energy_col_idx


class TrainingModel:
    type = ""
    name = ""
    columns = []
    _datas = None
    datas = None
    norm_datas = None
    window = None
    model = None

    val_perfor = None
    test_perfor = None
    predicts_list = None
    val_predicts_list = None


class BasicModel(TrainingModel):
    def __repr__(self):
        IPython.display.clear_output()
        return "###### {} Model Information ######\n".format(self.type) +\
            " - name : {}\n".format(self.name) +\
            " - columns : {}\n".format(reduce(lambda acc,
                                              cur: acc + "," + cur, self.columns)) +\
            " - datas : \n{}\n".format(self.datas['train'][:5]) +\
            " - norm_datas : \n{}\n".format(
                self.norm_datas['train'][:5])

    def __init__(self, name="",
                 columns=DEFAULT_COLUMNS.copy(),
                 is_switch=False,
                 is_contain_cluster_label=False):
        print("###### [Notice] {} model Init Start ###### \n".format(name))
        IPython.display.clear_output()
        self.energy_idx = col_check(columns)
        self.type = "Basic"
        self.name = name
        self.columns = columns

        # Data In
        db = KETI_DB()

        print("###### [Notice] {} model Init Start ###### \n".format(name))
        print("###### [Notice] jg datas load start ###### \n")
        self._datas = db.get_jungang_table()[columns]
        print(self._datas)
        print("\n###### [Notice] jg datas load success ###### \n")

        print("###### [Notice] Train, Val, Test Datas Config ###### \n")
        # Normalization Start
        day_1_size = 24
        year_half_size = day_1_size * int(365 / 4)
        year_1_size = day_1_size * 365

        train_datas = self._datas[:year_1_size]
        val_datas = self._datas[year_1_size:
                                year_1_size + year_half_size]
        test_datas = self._datas[year_1_size + year_half_size:]
        if is_switch == True:
            tmp = val_datas.copy()
            val_datas = test_datas.copy()
            test_datas = tmp

        print("###### [Notice] Normalization start ###### \n")
        mean = train_datas.mean()
        std = train_datas.std()

        print(mean)
        print(std)

        norm_train_datas = (train_datas - mean) / std
        norm_val_datas = (val_datas - mean) / std
        norm_test_datas = (test_datas - mean) / std
        print("\n###### [Notice] Normalization success ###### \n")

        if is_contain_cluster_label == True:
            print("###### [Notice] cluster datas load start ###### \n")
            cm = CLUSTER_MATCHING(self._datas)
            print("###### [Notice] cluster datas load success ###### \n")

            print("###### [Notice] cluster pattern matching start ###### \n")
            cluster_pattern_col = pd.DataFrame(columns=['cluster energy'])
            for idx in range(0, len(self._datas), 24):
                datas = self._datas[idx: idx + 24].copy()
                date = datas.index[0]

                c_pattern = cm.matching(date)

                for idx, _ in enumerate(datas.index):
                    cluster_pattern_col.loc[_] = c_pattern[idx]
            print(cluster_pattern_col)

            m = mean.values[self.energy_idx]
            s = std.values[self.energy_idx]

            norm_cluster_pattern_col = (cluster_pattern_col - m) / s
            norm_train_datas['cluster energy'] = norm_cluster_pattern_col['cluster energy']
            norm_val_datas['cluster energy'] = norm_cluster_pattern_col['cluster energy']
            norm_test_datas['cluster energy'] = norm_cluster_pattern_col['cluster energy']

            print(
                "\n###### [Notice] cluster pattern matching success ###### \n")

            del cm

        self.datas = {
            "train": train_datas,
            "val": val_datas,
            "test": test_datas,
        }
        self.norm_datas = {
            "train": norm_train_datas,
            "val": norm_val_datas,
            "test": norm_test_datas,
        }

        del db

    def set_window(self, WINDOW_WIDTH=3):
        # IPython.display.clear_output()
        print("###### [Notice] generate window start ###### \n")
        self.window = WindowGenerator(
            input_width=WINDOW_WIDTH,
            label_width=1,
            shift=1,
            label_columns=PREDICT_COLUMNS,
            train_df=self.norm_datas['train'],
            val_df=self.norm_datas['val'],
            test_df=self.norm_datas['test']
        )

        print(self.window)

        print("\n###### [Notice] generate window success ###### \n")

    def set_model(self, layer=64):
        # IPython.display.clear_output()
        print("###### [Notice] set lstm model start ###### \n")

        self.model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(
                layer, return_sequences=True, activation="tanh"),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(
                units=1
            )
        ])
        print(self.model)

        print("\n###### [Notice] set lstm model success ###### \n")

    def training(self, epochs=50):
        if self.window == None:
            raise SET_WINDOW_PLEASE
        if self.model == None:
            raise SET_MODEL_PLEASE
        # IPython.display.clear_output()
        compile_and_fit(self.model, self.window, EPOCHS=epochs)

        self.val_perfor = self.model.evaluate(self.window.val)
        self.test_perfor = self.model.evaluate(self.window.test)

    def plot(self, max_subplots=3):
        if self.window == None:
            raise SET_WINDOW_PLEASE

        self.window.plot(max_subplots=max_subplots)

    def get_original_pattern(self, is_reshape=False, is_val_datas=False):
        if is_val_datas == True:
            og_pattern = self.norm_datas['val']['energy (kw 15min)'].values
        else:
            og_pattern = self.norm_datas['test']['energy (kw 15min)'].values

        if is_reshape == True:
            return og_pattern.reshape(-1, 24)
        return og_pattern

    def statistic(self, predict_data_length=3, is_val_datas=False):
        if is_val_datas == False and self.predicts_list is None:
            raise SET_PREDICT_PLEASE
        if is_val_datas == True and self.val_predicts_list is None:
            raise SET_PREDICT_PLEASE

        if is_val_datas == True:
            predicts_list = self.val_predicts_list
        else:
            predicts_list = self.predicts_list
        statistic_datas = dict()

        og_pattern = self.get_original_pattern(
            is_reshape=True, is_val_datas=is_val_datas)
        mean_pattern = og_pattern.mean(axis=0)

        # Clustering Data Operator
        tss = 0
        for og in og_pattern:
            tss += euc_dis(
                mean_pattern[predict_data_length:],
                og[predict_data_length:]
            ) ** 2
        statistic_datas['tss'] = tss

        wss = 0
        distances = 0
        similarities = 0
        for idx, p_pattern in enumerate(predicts_list):
            distance = euc_dis(
                p_pattern,
                og_pattern[idx][predict_data_length:]
            )
            similarity = cos_sim(
                p_pattern,
                og_pattern[idx][predict_data_length:]
            )
            wss += distance ** 2

            distances = np.append(distances, [
                distance
            ])
            similarities = np.append(similarities, [
                similarity
            ])

        ecv = (1 - (wss / tss)) * 100
        mean_dis = distances.mean()
        mean_sim = similarities.mean()
        statistic_datas['wss'] = wss
        statistic_datas['ecv'] = ecv
        statistic_datas['mean dis'] = mean_dis
        statistic_datas['mean sim'] = mean_sim

        org_y = og_pattern[:, 3:].copy().flatten()
        pred_y = predicts_list.flatten()

        # mse
        mse = tf.keras.metrics.MeanSquaredError()
        mse.update_state(org_y, pred_y)

        # mae
        mae = tf.keras.metrics.MeanAbsoluteError()
        mae.update_state(org_y, pred_y)

        statistic_datas['mse'] = mse.result().numpy()
        statistic_datas['mae'] = mae.result().numpy()

        return statistic_datas

    def set_predict(self, is_reshape=False, predict_data_length=3, is_val_datas=False):
        if self.window == None:
            raise SET_WINDOW_PLEASE
        if self.model == None:
            raise SET_MODEL_PLEASE
        IPython.display.clear_output()
        print("###### [Notice] ({}) set predict ({}) info start ###### \n".
              format(self.name, "validation" if is_val_datas == True else "test"))

        predicts_list = np.array([])
        if is_val_datas == True:
            test_df = self.norm_datas['val'].copy()
        else:
            test_df = self.norm_datas['test'].copy()

        feature_length = len(test_df.columns)
        cnt = 0

        for split in range(0, round(len(test_df)), 24):
            if cnt % 50 == 0:
                print("{} / {}".format(cnt, round(len(test_df) / 24)))
            predicts = []

            for idx in range(0, (24 - predict_data_length)):
                inputs = test_df[split:(
                    split + 24)].values[idx: predict_data_length + idx].flatten()
                inputs = inputs.reshape(-1,
                                        predict_data_length, feature_length)
                result = self.model(inputs).numpy().flatten()[2]

                predicts.append(result)

            predicts_list = np.append(predicts_list, [predicts])
            cnt += 1
        print("{} / {} complete.".format(cnt, round(len(test_df) / 24)))

        if is_reshape == True:
            predicts_list = predicts_list.reshape(
                -1, 24 - predict_data_length)

        if is_val_datas == True:
            self.val_predicts_list = predicts_list.copy()
        else:
            self.predicts_list = predicts_list.copy()

        print("\n###### [Notice] set predict info success ###### \n")

    def plot_performance(self):
        IPython.display.clear_output()

        x = np.arange(1)
        width = 0.3

        metric_name = 'mean_absolute_error'
        metric_index = self.model.metrics_names.index(metric_name)
        val_mae = self.val_perfor[metric_index]
        test_mae = self.test_perfor[metric_index]

        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.ylabel(f'MAE (average over all times and outputs)')
        _ = plt.legend()
        plt.show()


class SeasonModel(TrainingModel):
    windows = None
    models = None

    def __repr__(self):
        IPython.display.clear_output()
        return "###### {} Model Information ######\n".format(self.type) +\
            " - name : {}\n".format(self.name) +\
            " - columns : {}\n".format(reduce(lambda acc,
                                              cur: acc + "," + cur, self.columns)) +\
            " - datas : \n{}\n".format(self.datas['train'][:5]) +\
            " - norm_datas : \n{}\n{}\n{}\n{}\n".format(
                self.norm_datas['train']['봄'][:5],
                self.norm_datas['train']['여름'][:5],
                self.norm_datas['train']['가을'][:5],
                self.norm_datas['train']['겨울'][:5])

    def __init__(self, name="",
                 columns=DEFAULT_COLUMNS.copy(),
                 is_switch=False,
                 is_contain_cluster_label=False):
        print("###### [Notice] {} model Init Start ###### \n".format(name))
        IPython.display.clear_output()
        if "season idx" not in columns:
            columns += ['season idx']
        self.energy_idx = col_check(columns)
        self.type = "Basic"
        self.name = name
        self.columns = columns

        # Data In
        db = KETI_DB()

        print("###### [Notice] {} model Init Start ###### \n".format(name))
        print("###### [Notice] jg datas load start ###### \n")
        self._datas = db.get_jungang_table()[columns]
        print(self._datas)
        print("\n###### [Notice] jg datas load success ###### \n")

        print("###### [Notice] Train, Val, Test Datas Config ###### \n")
        # Normalization Start
        day_1_size = 24
        year_half_size = day_1_size * int(365 / 4)
        year_1_size = day_1_size * 365

        train_datas = self._datas[:year_1_size]
        val_datas = self._datas[year_1_size:
                                year_1_size + year_half_size]
        test_datas = self._datas[year_1_size + year_half_size:]
        if is_switch == True:
            tmp = val_datas.copy()
            val_datas = test_datas.copy()
            test_datas = tmp

        # print("train", train_datas)
        # print("validation", val_datas)
        # print("test", test_datas)

        print("###### [Notice] Normalization start ###### \n")

        norm_train_dict = dict()
        norm_season_val_datas = pd.DataFrame()
        norm_season_test_datas = pd.DataFrame()
        norm_bak = dict()
        for season in SEASONS:
            print("###### [Notice] SEASON {} ###### \n".format(season))
            season_train_datas = train_datas[train_datas['season idx'] == get_season_to_idx(
                season)][train_datas.columns.difference(['season idx'])]
            season_val_datas = val_datas[val_datas['season idx'] == get_season_to_idx(
                season)][val_datas.columns.difference(['season idx'])]
            season_test_datas = test_datas[test_datas['season idx'] == get_season_to_idx(
                season)][test_datas.columns.difference(['season idx'])]

            # [train_datas\
            #     .columns.difference(['season_idx'])]
            # season 안에서만
            mean = season_train_datas.mean()
            std = season_train_datas.std()

            norm_train_datas = (season_train_datas - mean) / std

            if len(season_val_datas) != 0:
                norm_val_datas = (season_val_datas - mean) / std
                norm_season_val_datas = pd.concat(
                    [norm_season_val_datas, norm_val_datas])
            if len(season_test_datas) != 0:
                norm_test_datas = (season_test_datas - mean) / std
                norm_season_test_datas = pd.concat(
                    [norm_season_test_datas, norm_test_datas])

            norm_train_dict[season] = norm_train_datas

            norm_bak[season] = {
                "mean": mean,
                "std": std
            }
        # print("train", norm_train_dict)
        # print("validation", norm_season_val_datas)
        # print("test", norm_season_test_datas)

        print("\n###### [Notice] Normalization success ###### \n")
        if is_contain_cluster_label == True:
            print("###### [Notice] cluster datas load start ###### \n")
            cm = CLUSTER_MATCHING(self._datas)
            print("###### [Notice] cluster datas load success ###### \n")

            print("###### [Notice] cluster pattern matching start ###### \n")
            cluster_pattern_col = pd.DataFrame(columns=['cluster energy'])
            for idx in range(0, len(self._datas), 24):
                datas = self._datas[idx: idx + 24].copy()
                date = datas.index[0]
                season = get_season(date.month)

                c_pattern = cm.matching(date)

                m = norm_bak[season]["mean"][self.energy_idx]
                s = norm_bak[season]["mean"][self.energy_idx]

                c_pattern = (c_pattern - m) / s

                for idx, _ in enumerate(datas.index):
                    cluster_pattern_col.loc[_] = {
                        "cluster energy": c_pattern[idx],
                    }
            print(cluster_pattern_col)

            for season in SEASONS:
                norm_train_dict[season]['cluster energy'] = cluster_pattern_col['cluster energy']

            norm_season_val_datas['cluster energy'] = cluster_pattern_col['cluster energy']
            norm_season_test_datas['cluster energy'] = cluster_pattern_col['cluster energy']
            print(
                "\n###### [Notice] cluster pattern matching success ###### \n")

            del cm

        self.datas = {
            "train": train_datas,
            "val": val_datas,
            "test": test_datas,
        }
        self.norm_datas = {
            "train": norm_train_dict,
            "val": norm_season_val_datas,
            "test": norm_season_test_datas
        }
        print(self.norm_datas)

        del db

    def set_window(self, WINDOW_WIDTH=3):
        windows = dict()

        # IPython.display.clear_output()
        print("###### [Notice] generate window start ###### \n")
        for season in SEASONS:
            windows[season] = WindowGenerator(
                input_width=WINDOW_WIDTH,
                label_width=1,
                shift=1,
                label_columns=PREDICT_COLUMNS,
                train_df=self.norm_datas['train'][season],
                val_df=self.norm_datas['val'],
                test_df=self.norm_datas['test']
            )

        self.windows = windows
        print(self.windows)

        print("\n###### [Notice] generate window success ###### \n")

    def set_model(self, layer=64):
        models = dict()

        # IPython.display.clear_output()
        print("###### [Notice] set lstm model start ###### \n")

        for season in SEASONS:
            models[season] = tf.keras.models.Sequential([
                # Shape [batch, time, features] => [batch, time, lstm_units]
                tf.keras.layers.LSTM(
                    layer, return_sequences=True, activation="tanh"),
                # Shape => [batch, time, features]
                tf.keras.layers.Dense(
                    units=1
                )
            ])
        self.models = models
        print(self.models)

        print("\n###### [Notice] set lstm model success ###### \n")

    def training(self, epochs=50):
        if self.windows == None:
            raise SET_WINDOW_PLEASE
        if self.models == None:
            raise SET_MODEL_PLEASE
        # IPython.display.clear_output()
        for season in SEASONS:
            print(
                "\n###### [Notice] SEASON {} Training Start!!###### \n".format(season))
            compile_and_fit(self.models[season],
                            self.windows[season], EPOCHS=epochs)

        # self.val_perfor = self.model.evaluate(self.window.val)
        # self.test_perfor = self.model.evaluate(self.window.test)

    def plot(self, max_subplots=3):
        if self.window == None:
            raise SET_WINDOW_PLEASE

        self.window.plot(max_subplots=max_subplots)

    def get_original_pattern(self, is_reshape=False, is_val_datas=False):
        if is_val_datas == True:
            og_pattern = self.norm_datas['val']['energy (kw 15min)'].values
        else:
            og_pattern = self.norm_datas['test']['energy (kw 15min)'].values

        if is_reshape == True:
            return og_pattern.reshape(-1, 24)
        return og_pattern

    def statistic(self, predict_data_length=3, is_val_datas=False):
        if is_val_datas == False and self.predicts_list is None:
            raise SET_PREDICT_PLEASE
        if is_val_datas == True and self.val_predicts_list is None:
            raise SET_PREDICT_PLEASE

        if is_val_datas == True:
            predicts_list = self.val_predicts_list
        else:
            predicts_list = self.predicts_list
        statistic_datas = dict()

        og_pattern = self.get_original_pattern(
            is_reshape=True, is_val_datas=is_val_datas)
        mean_pattern = og_pattern.mean(axis=0)

        # Clustering Data Operator
        tss = 0
        for og in og_pattern:
            tss += euc_dis(
                mean_pattern[predict_data_length:],
                og[predict_data_length:]
            ) ** 2
        statistic_datas['tss'] = tss

        wss = 0
        distances = 0
        similarities = 0
        for idx, p_pattern in enumerate(predicts_list):
            distance = euc_dis(
                p_pattern,
                og_pattern[idx][predict_data_length:]
            )
            similarity = cos_sim(
                p_pattern,
                og_pattern[idx][predict_data_length:]
            )
            wss += distance ** 2

            distances = np.append(distances, [
                distance
            ])
            similarities = np.append(similarities, [
                similarity
            ])

        ecv = (1 - (wss / tss)) * 100
        mean_dis = distances.mean()
        mean_sim = similarities.mean()
        statistic_datas['wss'] = wss
        statistic_datas['ecv'] = ecv
        statistic_datas['mean dis'] = mean_dis
        statistic_datas['mean sim'] = mean_sim

        org_y = og_pattern[:, 3:].copy().flatten()
        pred_y = predicts_list.flatten()

        # mse
        mse = tf.keras.metrics.MeanSquaredError()
        mse.update_state(org_y, pred_y)

        # mae
        mae = tf.keras.metrics.MeanAbsoluteError()
        mae.update_state(org_y, pred_y)

        statistic_datas['mse'] = mse.result().numpy()
        statistic_datas['mae'] = mae.result().numpy()

        return statistic_datas

    def set_predict(self, is_reshape=False, predict_data_length=3, is_val_datas=False):
        if self.windows == None:
            raise SET_WINDOW_PLEASE
        if self.models == None:
            raise SET_MODEL_PLEASE
        IPython.display.clear_output()
        print("###### [Notice] ({}) set predict ({}) info start ###### \n".
              format(self.name, "validation" if is_val_datas == True else "test"))

        predicts_list = np.array([])
        if is_val_datas == True:
            test_df = self.norm_datas['val'].copy()
        else:
            test_df = self.norm_datas['test'].copy()

        print(test_df)
        feature_length = len(test_df.columns)
        cnt = 0

        for split in range(0, round(len(test_df)), 24):
            if cnt % 50 == 0:
                print("{} / {}".format(cnt, round(len(test_df) / 24)))
            predicts = []

            for idx in range(0, (24 - predict_data_length)):
                date = test_df.index[split]
                season = get_season(date.month)

                inputs = test_df[split:(
                    split + 24)].values[idx: predict_data_length + idx].flatten()
                inputs = inputs.reshape(-1,
                                        predict_data_length, feature_length)
                result = self.models[season](inputs).numpy().flatten()[2]

                predicts.append(result)

            predicts_list = np.append(predicts_list, [predicts])
            cnt += 1
        print("{} / {} complete.".format(cnt, round(len(test_df) / 24)))

        if is_reshape == True:
            predicts_list = predicts_list.reshape(
                -1, 24 - predict_data_length)

        if is_val_datas == True:
            self.val_predicts_list = predicts_list.copy()
        else:
            self.predicts_list = predicts_list.copy()

        print("\n###### [Notice] set predict info success ###### \n")

    def plot_performance(self):
        IPython.display.clear_output()

        x = np.arange(1)
        width = 0.3

        metric_name = 'mean_absolute_error'
        metric_index = self.model.metrics_names.index(metric_name)
        val_mae = self.val_perfor[metric_index]
        test_mae = self.test_perfor[metric_index]

        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.ylabel(f'MAE (average over all times and outputs)')
        _ = plt.legend()
        plt.show()


class ClusterModel(BasicModel):
    def __init__(self, name="", columns=DEFAULT_COLUMNS.copy(), is_switch=False):
        super().__init__(name=name, columns=columns,
                         is_switch=is_switch, is_contain_cluster_label=True)

    def set_window(self, WINDOW_WIDTH=3):
        print("###### [Notice] generate cluster window start ###### \n")
        self.window = CWindowGenerator(
            input_width=WINDOW_WIDTH,
            label_width=1,
            shift=1,
            label_columns=PREDICT_COLUMNS,
            train_df=self.norm_datas['train'],
            val_df=self.norm_datas['val'],
            test_df=self.norm_datas['test']
        )

        print(self.window)

        print("\n###### [Notice] generate cluster window success ###### \n")

    def set_predict(self, is_reshape=False, predict_data_length=3, is_val_datas=False):
        if self.window == None:
            raise SET_WINDOW_PLEASE
        if self.model == None:
            raise SET_MODEL_PLEASE
        IPython.display.clear_output()
        print("###### [Notice] ({}) set predict ({}) info start ###### \n".
              format(self.name, "validation" if is_val_datas == True else "test"))

        predicts_list = np.array([])
        if is_val_datas == True:
            test_df = self.norm_datas['val'][self.norm_datas['val'].columns.difference(
                ["cluster energy"])].copy()
        else:
            test_df = self.norm_datas['test'][self.norm_datas['test'].columns.difference([
                                                                                         "cluster energy"])].copy()

        feature_length = len(test_df.columns)
        cnt = 0

        for split in range(0, round(len(test_df)), 24):
            if cnt % 50 == 0:
                print("{} / {}".format(cnt, round(len(test_df) / 24)))
            predicts = []

            for idx in range(0, (24 - predict_data_length)):
                inputs = test_df[split:(
                    split + 24)].values[idx: predict_data_length + idx].flatten()
                inputs = inputs.reshape(-1,
                                        predict_data_length, feature_length)
                result = self.model(inputs).numpy().flatten()[2]

                predicts.append(result)

            predicts_list = np.append(predicts_list, [predicts])
            cnt += 1
        print("{} / {} complete.".format(cnt, round(len(test_df) / 24)))

        if is_reshape == True:
            predicts_list = predicts_list.reshape(
                -1, 24 - predict_data_length)

        if is_val_datas == True:
            self.val_predicts_list = predicts_list.copy()
        else:
            self.predicts_list = predicts_list.copy()

        print("\n###### [Notice] set predict info success ###### \n")

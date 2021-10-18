from src.util import *
from functools import reduce
import IPython
import IPython.display

DEFAULT_COLUMNS = [
    'energy (kw 15min)',
    'week sin',
    'week cos',
    'year sin',
    'year cos',
    'season idx',
    'avg ta',
    'avg rhm'
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

    def __repr__(self):
        IPython.display.clear_output()
        return "###### {} Model Information ######\n".format(self.type) +\
            " - name : {}\n".format(self.name) +\
            " - columns : {}\n".format(reduce(lambda acc,
                                              cur: acc + "," + cur, self.columns)) +\
            " - datas : \n{}\n".format(self.datas['train'][:5]) +\
            " - norm_datas : \n{}\n".format(self.norm_datas['train'][:5])


class BasicModel(TrainingModel):
    def __init__(self, name="",
                 columns=DEFAULT_COLUMNS.copy(),
                 is_switch=False,
                 is_contain_cluster_label=False):
        IPython.display.clear_output()
        self.energy_idx = col_check(columns)
        self.type = "Basic"
        self.name = name
        self.columns = columns

        # Data In
        db = KETI_DB()

        print("###### [Notice] jg datas load start ###### \n")
        self._datas = db.get_jungang_table()[columns]
        print(self._datas)
        print("\n###### [Notice] jg datas load success ###### \n")

        print("###### [Notice] Train, Val, Test Datas Config ###### \n")
        # Normalization Start
        day_1_size = 24
        year_half_size = day_1_size * int(365 / 4)
        year_1_size = day_1_size * 365

        is_switch = True

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

    def plot_performance(self):
        IPython.display.clear_output()

        x = np.arange(1)
        width = 0.3

        metric_name = 'mean_absolute_error'
        metric_index = self.model.metrics_names.index(metric_name)
        print(metric_index)
        val_mae = self.val_perfor[metric_index]
        test_mae = self.test_perfor[metric_index]

        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.ylabel(f'MAE (average over all times and outputs)')
        _ = plt.legend()
        plt.show()

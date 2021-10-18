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


def col_check(columns):
    for col in columns:
        if col not in DEFAULT_COLUMNS:
            return False
    return True


class TrainingModel:
    type = ""
    name = ""
    columns = []
    _datas = None
    datas = None
    norm_datas = None

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
        if len(columns) == 0:
            raise EMPTY_COL_EXCEPTION
        if col_check(columns) == False:
            raise INVALID_COL_EXCEPTION

        energy_col_idx = columns.index("energy (kw 15min)")
        if energy_col_idx == -1:
            raise ESSENTIAL_ENERGY_COL_EXCEPTION

        print("energy_col_idx : {}".format(energy_col_idx))

        self.type = "Basic"
        self.name = name
        self.columns = columns
        self.energy_idx = energy_col_idx

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
        print("###### [Notice] Normalization success ###### \n")

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

            m = mean.values[energy_col_idx]
            s = std.values[energy_col_idx]

            norm_cluster_pattern_col = (cluster_pattern_col - m) / s
            norm_train_datas['cluster energy'] = norm_cluster_pattern_col['cluster energy']
            norm_val_datas['cluster energy'] = norm_cluster_pattern_col['cluster energy']
            norm_test_datas['cluster energy'] = norm_cluster_pattern_col['cluster energy']

            print(
                "\n###### [Notice] cluster pattern matching success ###### \n")

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

        del cm
        del db

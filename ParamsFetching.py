from math import sqrt

import numpy as np
import yaml



class ParamsFetching:

    def __init__(self):
        with open("config/pre_strata_matrix.yml", "r", encoding="utf8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
                            # africa    australia   europe  n-asia   s-america  s-asia n-america
        self.stratum_dict={1:"dbt_af", 2:"ebt_au", 3:"dbt_eu", 4:"dbt", 5:"ebt_sa", 6:"ebt", 7:"dbt_na",#DBT
                           8:"ebt_af", 9:"ebt_au", 10:"dbt_eu", 11: "dbt", 12: "ebt_sa", 13: "ebt", 14: "dbt_na",#EBT
                           15: "ent", 16: "ent_au", 17: "ent_eu", 18: "ent", 19: "ent", 20: "ent", 21: "ent_na",#ENT
                           22: "ent", 23: "ent_au", 24: "ent_eu", 25: "ent", 26: "ent", 27: "ent", 28: "ent_na",#DNT
                           29: "gsw", 30: "gsw_auoc", 31: "gsw", 32: "gsw", 33: "gsw", 34: "gsw", 35: "gsw"#GSW
                           }
        self.rh_index_dict = {40: 1, 50: 2, 60: 3, 70: 4, 98: 5, 0: -1}

    def get_params(self, stratum_num):
        params = self.config.get(self.stratum_dict[stratum_num])
        return params

    def get_agbd(self, array):
        agbd = np.zeros((array.shape[0], array.shape[1])).astype(np.float32)
        ps_array = array[:, :, 0]
        ps_list = np.unique(ps_array).astype(int).tolist()
        for ps_id in ps_list:
            if ps_id == 241 or ps_id not in self.stratum_dict.keys():
                continue
            params = self.config.get(self.stratum_dict[ps_id])
            rh1 = params.get('c_rh')
            rh2 = params.get('d_rh')
            i1 = self.rh_index_dict[rh1]+1
            rh1_masked_by_ps = np.where(ps_array == ps_id, array[:, :, i1], np.nan)
            if rh2 != 0:
                i2 = self.rh_index_dict[rh2] + 1
                rh2_masked_by_ps = np.where(ps_array == ps_id, array[:, :, i2], np.nan)
                if ps_id=='gsw_auoc':
                    rh3 = params.get('e_rh')
                    i3 = self.rh_index_dict[rh3] + 1
                    rh3_masked_by_ps = np.where(ps_array == ps_id, array[:, :, i3], np.nan)
                    agbd_ps = params.get('a') * pow(params.get('b') +
                                                    params.get('c') * np.sqrt(rh1_masked_by_ps+100) +
                                                    params.get('d') * np.sqrt(rh2_masked_by_ps+100) +
                                                    params.get('e') * np.sqrt(rh3_masked_by_ps+100), 2) /100
                else:
                    agbd_ps = params.get('a') * pow(params.get('b') + params.get('c') * np.sqrt(rh1_masked_by_ps+100) +
                                                   params.get('d') * np.sqrt(rh2_masked_by_ps+100), 2) /100
            else:
                agbd_ps = params.get('a') * pow(params.get('b') + params.get('c') * np.sqrt(rh1_masked_by_ps + 100), 2) /100
            agbd += np.where(np.isnan(agbd_ps), 0, agbd_ps)
            agbd = np.where(agbd==0, -1, agbd)
        return agbd

if __name__=='__main__':
    params_fetching = ParamsFetching()
    params=params_fetching.get_agbd(np.ones((256,256,6)))
    print(params)
    # import tensorflow as tf
    # import tensorflow.python.keras.backend as K
    #
    #
    # def masked_mse(y_true, y_pred):
    #     y_true = tf.reshape(y_true, -1)
    #     y_pred = tf.reshape(y_pred, -1)
    #     mask_true = K.cast(K.not_equal(y_true, -1), K.floatx())
    #     masked_squared_error = K.square(mask_true * (y_true - y_pred))
    #     masked_mse = K.mean(K.sum(masked_squared_error, axis=-1) / (K.sum(mask_true, axis=-1) + K.epsilon()))
    #     return masked_mse
    #
    #
    # print(masked_mse(array[:, :, 1] / 100, agbd))


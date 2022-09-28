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
        agbd = np.zeros((256, 256))
        ps_array = array[:,:,0]
        ps_list = np.unique(ps_array).astype(int).tolist()
        for ps_id in ps_list:
            if ps_id == 241 or ps_id not in self.stratum_dict.keys():
                array_masked_by_ps = np.where(array == ps_id, np.nan, 0)
                agbd += array_masked_by_ps[:, :, 0]
                continue
            array_masked_by_ps = np.where(array == ps_id, array, 0)
            params = self.config.get(self.stratum_dict[ps_id])
            rh1 = params.get('c_rh')
            rh2 = params.get('d_rh')
            if rh2 != 0:
                i1 = self.rh_index_dict[rh1]
                i2 = self.rh_index_dict[rh2]
                agbd += params.get('a') * pow((params.get('b') + params.get('c') * np.sqrt(array_masked_by_ps[:, :, i1]+100) +
                                               params.get('d') * np.sqrt(array_masked_by_ps[:, :, i2]+100)), 2)
            else:
                i1 = self.rh_index_dict[rh1]
                agbd += params.get('a') * pow(params.get('b') + params.get('c') * np.sqrt(array_masked_by_ps[:, :, i1] + 100), 2)
        return agbd

if __name__=='__main__':
    params_fetching = ParamsFetching()
    params=params_fetching.get_agbd(np.ones((256,256,6)))
    print(params)
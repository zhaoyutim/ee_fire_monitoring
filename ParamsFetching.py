from math import sqrt

import yaml



class ParamsFetching:

    def __init__(self):
        with open("config/pre_strata_matrix.yml", "r", encoding="utf8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
                            # africa    australia   europe  n-asia   s-america  s-asia n-america
        self.stratum_dict={1:"dbt_af", 2:"ebt_au", 3:"dbt_eu", 4:"dbt", 5:"ebt_sa", 6:"ebt", 7:"dbt_na",#DBT
                           8:"ebt_af", 9:"ebt_au", 10:"dbt_eu", 11: "dbt", 12: "ebt_sa", 13: "ebt", 14: "dbt_na",#EBT
                           15: "n/a", 16: "ent_au", 17: "ent_eu", 18: "ent", 19: "n/a", 20: "ent", 21: "ent_na",#DNT
                           22: "ent", 23: "ent_au", 24: "ent_eu", 25: "ent", 26: "ent", 27: "ent", 28: "ent_na",#ENT
                           29: "gsw", 30: "gsw_auoc", 31: "gsw", 32: "gsw", 33: "gsw", 34: "gsw", 35: "gsw"#GSW
                           }

    def get_params(self, stratum_num):
        params = self.config.get(self.stratum_dict[stratum_num])
        return params

    def get_agbd(self, stratum_num):
        params = self.config.get(self.stratum_dict[stratum_num])
        rh1 = params.get('c_rh')
        rh2 = params.get('d_rh')
        agbd =  params.get('a') * pow((params.get('b') + params.get('c') * sqrt(rh1+100) + params.get('d') * sqrt(rh2+100)), 2)

if __name__=='__main__':
    params_fetching = ParamsFetching()
    params=params_fetching.get_params(1)
    print(params)

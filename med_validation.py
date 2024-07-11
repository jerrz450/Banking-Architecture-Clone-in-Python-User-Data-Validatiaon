import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Dict
import sys
import os
from db_sql import df
import re
import heapq

class FakeOrReal:

    def __init__(self) -> None:
        self.memo = {}
      
    def min_dis(self, target_s, source_s):
        if source_s ==  target_s:
            return [target_s, 0]
    
        try: 
            source1 = list(source_s)
            target1 = list(target_s)

        except TypeError:
            return [target_s, None]

        s1 = list(range(len(source1) + 1))
        t1 = list(range(len(target1) + 1))

        matrix = np.zeros((len(s1), len(t1)))
        matrix[0, :] = t1
        matrix[:, 0] = s1

        for r in range(1, len(s1)):
            for c in range(1, len(t1)):
                key = (tuple(source1[:r]), tuple(target1[:c]))
                if key in self.memo:
                    matrix[r, c] = self.memo[key]
                else:
                    if target1[c - 1] != source1[r - 1]:
                        matrix[r, c] = min(matrix[r - 1, c], matrix[r, c - 1], matrix[r - 1, c - 1]) + 1
                    else:
                        matrix[r, c] = matrix[r - 1, c - 1]
                    self.memo[key] = matrix[r, c]


        return [target_s, int(matrix[-1, -1])]

    def run_data_slo(self, target, data_to_search):
        # tgt = ["ontario", "orleans"]
        # sour = ['east toronto', 'india bazaar', 'the beaches west']
        data1 = {}

        if target:
            
            target_list = target if isinstance(target, list) else [target]
            target_list = [s.strip() for s in target_list if s.strip()]

            for key, source in data_to_search.items():

                sources = re.split(r'[()|/]', source) if re.findall(r'[()|/]', source) else [source]
                sources1 = [s.strip() for s in sources if s.strip()]

                value = []

                if all(sources1[x] == target_list[x] for x in range(len(target_list))):
                    value.append(0)
                    unique_key = f"{', '.join(sources1)}"
                    data1[unique_key] = value
                    continue
                
                for sour in sources1:
                    for tgt in target_list:
                        distance_data = self.min_dis(target_s = tgt.strip(), source_s = sour.strip())
                        value.append(distance_data[1])

                unique_key = f"{', '.join(sources1)}"

                data1[unique_key] = value
        

        new =  {k : sum(heapq.nsmallest(2 if len(v) != 1 else 1, v)) for k, v in data1.items()}
        key = list(new.keys())[list(new.values()).index(min(new.values()))]
        
        # print(key)
        # new  = {k : float("{0: .2f}".format(sum(v) / len(v))) for k, v in data1.items()}
        # key = list(new.keys())[list(new.values()).index(min(new.values()))]
        # print(key)
        if not data1:
            raise ValueError("No valid distance data found.")
        
        min_value =  min(new.values())
        new =  {k : sum(heapq.nsmallest(2 if len(v) != 1 else 1, v)) for k, v in data1.items()}
        key = list(new.keys())[list(new.values()).index(min_value)]

        print(data1)
        threshold = 3

        if min_value > threshold:
            print(f"{target} is unfortunately not correct, did you maybe mean -> {key.capitalize()}?")
            return target  # Target word is flagged as potentially fake.
        else:
            return target  # Target word is considered valid.
       
    def main_run(self, data_to_search, user_data):
        
        def contains_non_alphanumeric(data):
            for key, value in data.items():
                if re.findall(r'[()|/]', value):
                    return True
                else:
                    return False
                
        def split_if_contains_ascii(value):
            if re.search(r'[()|/]', value):
                return list(filter(None, re.split(r'[()|/]', value)))
            else:
                return value
                        
        if contains_non_alphanumeric(user_data):
           user_data = {k: split_if_contains_ascii(v) for k, v in user_data.items()}

        
        if user_data["County"] is None and user_data["State"] is None:
            data_to_dict = [(data_to_search["postalCode"], user_data["Post Code"].strip()), 
                    (data_to_search["placeName"], user_data["City"])]
            
            keys = ["postalCode", "placeName"]

        else:
            data_to_dict = [(data_to_search["postalCode"], user_data["Post Code"].strip()), 
            (data_to_search["placeName"], user_data["City"]),
            (data_to_search["adminName2"], user_data["County"]), 
            (data_to_search["adminName1"], user_data["State"])]
          
            keys = ["postalCode", "placeName", "adminName2", "adminName1"]

        with ThreadPoolExecutor() as executor:
            tasks = {executor.submit(self.run_data_slo, arg2, arg1): key for key, (arg1, arg2) in zip(keys, data_to_dict)}
            final_value = [{key: task.result()} for task, key in tasks.items()]
        
        dict_new = {}
        [dict_new.update(value) for value in final_value]
        print(dict_new)
        return dict_new


if __name__ == "__main__":
    pass
    # dict_data_user = df.to_dict()
    # main = FakeOrReal().main_run(data_to_search= dict_data_user, user_data= {"City" : "ofc", "Post Code" : "m3k", "County" : "toronto", "State" : "ontario"}) 
    # print(main)

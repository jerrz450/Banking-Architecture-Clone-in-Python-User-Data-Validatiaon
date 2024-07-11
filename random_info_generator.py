import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

class FakeOrReal:

    def __init__(self, data_to_search, user_data) -> None:
        self.memo = {}
        self.user_data =  user_data
        self.data_to_search = data_to_search


    def min_dis(self, target_s, source_s):
        source1 = list(source_s)
        target1 = list(target_s)
        
        s1 = list(range(len(source1) + 1))
        t1 = list(range(len(target1) + 1))

        matrix = np.zeros((len(s1), len(t1)))
        matrix[0, :] = t1
        matrix[:, 0] = s1

        for r in range(1, len(s1)):
            for c in range(1, len(t1)):
                key = (source_s[:r], target_s[:c])
                if key in self.memo:
                    matrix[r, c] = self.memo[key]
                else:
                    if target1[c - 1] != source1[r - 1]:
                        matrix[r, c] = min(matrix[r - 1, c], matrix[r, c - 1], matrix[r - 1, c - 1]) + 1
                    else:
                        matrix[r, c] = matrix[r - 1, c - 1]
                    self.memo[key] = matrix[r, c]
        return [target_s, int(matrix[-1, -1])]

    def run_data_slo(self):
        target = self.user_data
        data = self.data_to_search

        dict_data = data
        data1 = {}

        if target:
            for source in dict_data.values():
                distance_data = self.min_dis(source_s= target, target_s= source)
                data1[distance_data[0]] = distance_data[1]

            edit_distance = min(data1.values())
            key = list(data1.keys())[list(data1.values()).index(edit_distance)]

            calculated_data = (edit_distance / max(len(key), len(target))) * 100
            threshold_percentage = 12

            if calculated_data > threshold_percentage:
                return (target, False)  # Target word is flagged as potentially fake.
            else:
                return (target, True)  # Target word is considered valid.
        else:
            return False

    def main_run(self):

        data_to_dict = (self.data_to_search["postalCode"], "6969"), (self.data_to_search["placeName"], "Vransko")

        with ThreadPoolExecutor() as executor:
            tasks = []
            for arg1, arg2 in data_to_dict:
                fake_or_real = FakeOrReal(user_data=arg2, data_to_search=arg1)
                tasks.append(executor.submit(fake_or_real.run_data_slo))

        for task in tasks:
            print(task.result())
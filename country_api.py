import asyncio
import aiohttp
import time
import json
import urllib.parse
from api_s import FindAMatch
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from med_validation import FakeOrReal
import sys
import re

class Get_Slo_Naslovi:
    def __init__(self, input : dict) -> None:
        # self.input = re.sub(r'[^a-zA-Z]', '', input).lower()
        self.input = input
        self.path = "C:\\Users\\Jernej\\Documents\\OOP\\database_files\\naslovi1.csv"

    async def separate_thread(self, function, argument):
        with ThreadPoolExecutor(1, "Gathering_Naslovi_Data") as executor:
            data = await asyncio.get_event_loop().run_in_executor(executor, function, argument)
            return data
            
    def min_ed_dist(self, data_set):
    
        data = FakeOrReal().main_run(user_data = self.input, data_to_search= data_set)
        return data
        
    async def run(self, dataset):
        result = asyncio.gather(self.separate_thread(self.min_ed_dist, dataset))
        return result

class GetData:
    def __init__(self, code) -> None:
        self.code = code

        self.headers = {
                'X-Parse-Application-Id': f'{os.environ.get("4_APP_API_ID")}', 
                'X-Parse-REST-API-Key': f'{os.environ.get("4_APP_API_PASS")}'
                        }
        self.url = f"https://parseapi.back4app.com/classes/Worldzipcode_{self.code}?limit=5000000000000000000"

    async def fetch_data_country(self):
        attempt = 0
        max_attempts = 3

        while attempt < max_attempts:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.url, headers=self.headers) as response:
                        response.raise_for_status()
                        data = await response.json()
                        df = pd.json_normalize(data, 'results')
                        df.drop(columns=['adminCode3', 'accuracy', 'createdAt', 'updatedAt', 'geoPosition.__type', 'geoPosition.latitude', 'geoPosition.longitude'], inplace=True, errors='ignore')
                        return df
                        
            except KeyError:
                attempt += 1
                time.sleep(1)  # Sleep before retrying
            except Exception:
                attempt += 1
                time.sleep(1)  # Sleep before retrying
        return None

class Vectorize_Dataframe:

    def __init__(self, df, city, postal_code, country, county=None, state=None) -> None:
        self.df = df
        self.postal_code = postal_code
        self.city = city
        self.country = country
        self.county = county or None
        self.state = state or None
            
    def validate_final_data(self, user_data: dict, df: pd.DataFrame):
        pd.set_option('future.no_silent_downcasting', True)

        # Preprocessing
        for column in df.columns:
            df[column] = df[column].fillna('')
            df[column] = np.where(df[column].notna() & df[column].str.contains(r"[()/]", regex=True), df[column].str.replace(r"[()/]", "-", regex=True), df[column])
            df[column] = np.where(df[column].notna(), df[column].str.strip().replace(r"[\W|\s|_]", "-", regex=True), df[column])
            df[column] = np.where(df[column].str.endswith('-'), df[column].str[:-1].str.strip(), df[column].str.strip())
            df[column] = df[column].apply(lambda x: x.split('-') if isinstance(x, str) else x)

        # Define matching function
        def match_values(series, user_value):
            return series.map(lambda x: any(user_value in item.lower() for item in x) if isinstance(x, list) else user_value in str(x).lower())

        # Clean user data
        def test(value):
            if re.search(r"[()|/]", value):
                return re.split(r"[()|/]", value)
            else:
                return value
            
        cleaned_user_data_dict = {k: test(v) for k, v in user_data.items()}
        print(cleaned_user_data_dict)
        """ dictionary.get(key, default_value)
            key: The key for which you want to retrieve the value.
            default_value: (Optional) The value to return if the key does not exist. If not provided, None is returned by default.
        """
        result = df
        [
            match_values(df['placeName'], cleaned_user_data_dict.get("placeName", '')) &
            match_values(df['postalCode'], cleaned_user_data_dict.get("postalCode", '')) |
            match_values(df['adminName1'], cleaned_user_data_dict.get("adminName1", '')) |
            match_values(df['adminName2'], cleaned_user_data_dict.get("adminName2", ''))
        ]

        result.reset_index(drop=True, inplace=True)
        
        if result.empty:
            return False
        
        else:
            values_dict = {}

            for column in result.columns:
                values_dict[column] = result[column].iloc[0] if len(result[column]) > 0 else ''
            return values_dict

    async def gather_df_data(self):
        
        """.astype() is a method within numpy.ndarray, as well as the Pandas Series class, so can be used to convert vectors, matrices and columns within a DataFrame. However, int() is a pure-Python function that can only be applied to scalar values.
            For example, you can do int(3.14), but can't do (2.7).astype('int'), because Python native types don't have any such method. However, numpy.array([1.1, 2.2, 3.3]).astype('int') is valid.
            (Strictly, it is also possible to define an __int__() method within one's own classes, which would allow int() to be applied to non-native types."""
        

        if self.county is not None or self.state is not None:

            self.df["placeName"] = self.df["placeName"].str.lower()
            self.df["adminName1"] = self.df["adminName1"].str.lower()
            self.df["adminName2"] = self.df["adminName2"].str.lower().fillna('')
            self.df["postalCode"] = self.df["postalCode"].str.lower()

            city = self.df["placeName"].str.lower().str.contains(self.city, regex = False, na=False)
            postal_code = self.df["postalCode"].str.contains(self.postal_code, na = False)
            state = self.df["adminName1"].str.lower().str.contains(self.state, na=False)
            county = self.df["adminName2"].str.lower().str.contains(self.county, na=False)

            # Assign weights
            city_weight = 2 #City has higher prioritization than others, so that the algorithm can find it sooner and more accurately.
            state_weight = 2
            county_weight = 2
            postal_code_weight = 3

            # Calculate weighted match count
            match_count = (city.astype(int) * city_weight +
                        state.astype(int) * state_weight +
                        county.astype(int) * county_weight +
                        postal_code.astype(int) * postal_code_weight)
            # Get top 80 matches
            top_match_indices = match_count.nlargest(80).index

            # print(np.where(self.df['postalCode'] == "k1e"))
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
        
            result_top_match_indices_df = pd.DataFrame(self.df.loc[top_match_indices]).drop(columns= ["objectId", "adminName3"])
            result_top_match_indices_dict = pd.DataFrame(self.df.loc[top_match_indices]).drop(columns= ["objectId", "adminName3"]).to_dict()

            while True:
                try:
                    render_naslov_task = await asyncio.create_task(Get_Slo_Naslovi(input = {"City" : self.city, "Post Code" : self.postal_code, "County" : self.county, "State" : self.state}).run(dataset = result_top_match_indices_dict))
                    rendered = await render_naslov_task
                    break 
                except asyncio.CancelledError:
                    print("Task was cancelled, retrying...")
                    time.sleep(1) 
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return None
                
            if rendered:
                data_validated = self.validate_final_data(rendered[0], result_top_match_indices_df)
                return data_validated
            
            return rendered[0] if rendered else None
    
        else:
            
            city = self.df["placeName"].str.contains(self.city, na=False)
            self.df['postalCode'] = self.df['postalCode'].str.lower()
            postal_code = self.df["postalCode"].str.contains(self.postal_code)
            self.df["placeName"] = self.df["placeName"].str.lower()

            # Assign weights
            city_weight = 3
            postal_code_weight = 2.5

            # Calculate weighted match count
            match_count = (city.astype(int) * city_weight + postal_code.astype(int) * postal_code_weight)
         
            # Get top 10 matches
            top_match_indices = match_count.nlargest(80).index
            result_top_match_indices = pd.DataFrame(self.df.loc[top_match_indices]).drop(columns= ["objectId", "adminName3"])
    
            if self.country == "SI":
                data_slo = pd.read_csv("C:\\Users\\Jernej\\Documents\\OOP\\database_files\\naslovi1.csv").to_dict()["Naslov"]
                data_slo_dict = {"placeName" : data_slo, "postalCode" : result_top_match_indices["postalCode"].to_dict()}
                
                while True:
                    try:
                        render_naslov_task = await asyncio.create_task(Get_Slo_Naslovi(input = {"City" : self.city, "Post Code" : self.postal_code, "County" : None, "State" : None}).run(dataset = data_slo_dict))
                        rendered = await render_naslov_task
                        break 
                    except asyncio.CancelledError:
                        print("Task was cancelled, retrying...")
                        time.sleep(1) 
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        return None       
                             
                if rendered:

                    data_validated = self.validate_final_data(rendered[0], result_top_match_indices)
                    if data_validated:
                        data_validated["Country"] = self.country
                    return data_validated
                
                else:
                    return rendered[0] if rendered else None
            
            else:
                while True:
                    
                    try:
                        render_naslov_task = await asyncio.create_task(Get_Slo_Naslovi(input = {"City" : self.city, "Post Code" : self.postal_code, "County" : None, "State" : None}).run(dataset = result_top_match_indices))
                        rendered = await render_naslov_task
                        break 

                    except asyncio.CancelledError:
                        print("Task was cancelled, retrying...")
                        time.sleep(1) 
                    
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        return None
         
    @staticmethod
    async def get_match(match, list_choices):
        matched = FindAMatch(value=match, choices=list_choices).find_match()
        return matched

    async def run_gather_df(self):
        data_df = await self.gather_df_data()
        return data_df

if __name__ == "__main__":

    start = time.perf_counter()
    data2 = asyncio.run(GetData("CA").fetch_data_country())
    data1 = asyncio.run(Vectorize_Dataframe(df = data2, country= "CA", city= "toronto", postal_code= "M5H 2N2", county= "toronto", state= "ontario").run_gather_df())
    result_df = data1[1]
    result = result_df[(result_df['placeName'] == "toronto") & (result_df['postalCode'] == "M5H") & (result_df['adminName1'] == "toronto")]
  
    end = time.perf_counter()
    print(f"Task ended in {end - start} seconds!")
    #data = GetData(city = city, postal_code = postal_code, code = country_code[1], county= None, state= None).run_zipcode_api_call()

def test():
    async def call_api(self):
        
        async with aiohttp.ClientSession() as session:
            tasks =  [self.get_whole(session, self.url)]
            results = await asyncio.gather(*tasks)
            list_data = [self.get_place_data(df = results, place = "Los Angeles")]

            whole_result = await asyncio.gather(*list_data)
            match_result = await self.get_match(match= self.place, list_choices= whole_result)

            if match_result[0]:
                task2 = [self.get_content(session, self.get_query(place = match_result[1]))]
                results2 = await asyncio.gather(*task2)
                one_result = await self.process_results_one(results2[0])
                return one_result

    def get_query(self, place, fields=["placeName", "adminName1", "adminName2", "adminName3"]):
        regex_pattern = f".*{place}.*"
        query = {"$or": [{field: {"$regex": regex_pattern, "$options": "i"}} for field in fields]}
        query1 = json.dumps(query)
        encoded = urllib.parse.quote_plus(query1)
        url = self.base_url + encoded
        return url

    async def get_content(self, session, url):
        async with session.get(url, headers= self.headers) as response:
            response.raise_for_status()
            data = await response.json()
            return dict(data)

    async def proccess_result_whole(self,result):
        list1 = [result[data] for data in ["placeName", "adminName1", "adminName2", "adminName3"] if result[data]]
        final_list = ''.join(list1)
        return final_list
        
    async def process_results_one(self, result):
        if not result['results']:
            return False
        else:
            sample_dict = result['results']
            return sample_dict
        
    async def get_place_data(self, df : pd.DataFrame):
        if self.county is not None or self.state is not None: 
            city = df["placeName"].str.contains("Chicago", na= False)
            state = df["adminName1"].str.contains("Illinois",na= False)
            county = df["adminName2"].str.contains("'Cook County", na= False)
            postal_code = df["postalCode"].str.contains("60601")

            match_count = city.astype(int) + state.astype(int) + county.astype(int) + postal_code.astype(int) 
            
            max_match_index = match_count.idxmax()

            return df.loc[max_match_index] if match_count[max_match_index] > 0 else False
        else:
            city = df["placeName"].str.contains(self.city, na= False)
            postal_code = df["postalCode"].str.contains(self.postal_code)

            match_count = city.astype(int) + postal_code.astype(int)

            max_match_index = match_count.idxmax()

            return df.loc[max_match_index] if match_count[max_match_index] > 0 else False

    async def get_match(self, match, list_choices):
        matched = FindAMatch(value= match, choices= list_choices).find_match()
        return matched
    
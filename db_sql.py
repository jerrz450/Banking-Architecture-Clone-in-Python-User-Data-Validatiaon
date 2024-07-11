import pandas as pd
from api_s import FindAMatch
import numpy as np
import asyncio
import time
import re
import nltk
import re
from Levenshtein import distance as levenshtein_distance
# Sample DataFrame
df = pd.DataFrame({'countryCode': {735: 'CA', 662: 'CA', 667: 'CA', 671: 'CA', 674: 'CA', 675: 'CA', 676: 'CA', 677: 'CA', 678: 'CA', 679: 'CA', 680: 'CA', 681: 'CA', 682: 'CA', 683: 'CA', 684: 'CA', 685: 'CA', 686: 'CA', 687: 'CA', 688: 'CA', 689: 'CA', 690: 'CA', 691: 'CA', 692: 'CA', 693: 'CA', 694: 'CA', 695: 'CA', 696: 'CA', 697: 'CA', 698: 'CA', 699: 'CA', 700: 'CA', 701: 'CA', 702: 'CA', 703: 'CA', 704: 'CA', 705: 'CA', 706: 'CA', 707: 'CA', 708: 'CA', 709: 'CA', 714: 'CA', 718: 'CA', 641: 'CA', 642: 'CA', 643: 'CA', 644: 'CA', 645: 'CA', 646: 'CA', 648: 'CA', 649: 'CA', 650: 'CA', 651: 'CA', 652: 'CA', 653: 'CA', 654: 'CA', 655: 'CA', 656: 'CA', 657: 'CA', 658: 'CA', 659: 'CA', 660: 'CA', 661: 'CA', 663: 'CA', 664: 'CA', 665: 'CA', 666: 'CA', 668: 'CA', 669: 'CA', 670: 'CA', 672: 'CA', 673: 'CA', 710: 'CA', 711: 'CA', 712: 'CA', 713: 'CA', 715: 'CA', 716: 'CA', 717: 'CA', 719: 'CA', 720: 'CA'}, 'postalCode': {735: 'k1e', 662: 'm3k', 667: 'm4k', 671: 'm4j', 674: 'm4e', 675: 'm4n', 676: 'm4m', 677: 'm4l', 678: 'm4r', 679: 'm4p', 680: 'm5a', 681: 'm4v', 682: 'm4t', 683: 'm5l', 684: 'm5r', 685: 'm4w', 686: 'm4y', 687: 'm5c', 688: 'm5g', 689: 'm4x', 690: 'm4s', 691: 'm5k', 692: 'm5e', 693: 'm5n', 694: 'm5s', 695: 'm5h', 696: 'm5j', 697: 'm5b', 698: 'm5t', 699: 'm5p', 700: 'm5v', 701: 'm6h', 702: 'm6j', 703: 'm6g', 704: 'm5w', 705: 'm6p', 706: 'm5x', 707: 'm6s', 708: 'm6k', 709: 'm7y', 714: 'm6r', 718: 'm8v', 641: 'l0n', 642: 'p0r', 643: 'l9e', 644: 'n3t', 645: 'n3p', 646: 'l8h', 648: 'n3r', 649: 'l8e', 650: 'n3s', 651: 'l0p', 652: 'k0h', 653: 'n3v', 654: 'l8g', 655: 'l8m', 656: 'l8l', 657: 'l8k', 658: 'l8j', 659: 'l8n', 660: 'l8r', 661: 'l8p', 663: 'l8v', 664: 'l9a', 665: 'l9b', 666: 'l9c', 668: 'l8t', 669: 'k0j', 670: 'l8s', 672: 'l8w', 673: 'k0l', 710: 'k1n', 711: 'k1a', 712: 'k1h', 713: 'k1g', 715: 'k1m', 716: 'k1l', 717: 'k1p', 719: 'k1k', 720: 'k1r'}, 'placeName': {735: 'orleans (queenswood)', 662: 'downsview east (cfb toronto)', 667: 'east toronto (the danforth west / riverdale)', 671: 'east toronto (the danforth east)', 674: 'east toronto (the beaches)', 675: 'central toronto (lawrence park east)', 676: 'east toronto (studio district)', 677: 'east toronto (india bazaar / the beaches west)', 678: 'central toronto (north toronto west)', 679: 'central toronto (davisville north)', 680: 'downtown toronto (regent park / port of toronto)', 681: 'central toronto (summerhill west / rathnelly / south hill / forest hill se / deer park)', 682: 'central toronto (moore park / summerhill east)', 683: 'downtown toronto (commerce court / victoria hotel)', 684: 'central toronto (the annex / north midtown / yorkville)', 685: 'downtown toronto (rosedale)', 686: 'downtown toronto (church and wellesley)', 687: 'downtown toronto (st. james park)', 688: 'downtown toronto (central bay street)', 689: 'downtown toronto (st. james town / cabbagetown)', 690: 'F', 691: 'downtown toronto (toronto dominion centre / design exchange)', 692: 'downtown toronto (berczy park)', 693: 'central toronto (roselawn)', 694: 'downtown toronto (university of toronto / harbord)', 695: 'downtown toronto (richmond / adelaide / king)', 696: 'downtown toronto (harbourfront east / union station / toronto island)', 697: 'downtown toronto (ryerson)', 698: 'downtown toronto (kensington market / chinatown / grange park)', 699: 'central toronto (forest hill north & west)', 700: 'downtown toronto (cn tower / king and spadina / railway lands / harbourfront west / bathurst quay / south niagara / ytz)', 701: 'west toronto (dufferin / dovercourt village)', 702: 'west toronto (rua aþores / trinity)', 703: 'downtown toronto (christie)', 704: 'downtown toronto stn a po boxes 25 the esplanade (enclave of m5e)', 705: 'west toronto (high park / the junction south)', 706: 'downtown toronto (underground city)', 707: 'west toronto (bloor west village / swansea)', 708: 'west toronto (brockton / parkdale village / exhibition place)', 709: 'east toronto business reply mail processing centre 969 eastern (enclave of m4l)', 714: 'west toronto (parkdale / roncesvalles village)', 718: 'etobicoke (new toronto / mimico south / humber bay shores)', 641: 'dufferin county (shelburne)', 642: 'algoma southwest (blind river)', 643: 'milton', 644: 'brantford southwest', 645: 'brantford northeast', 646: 'hamilton (west kentley / mcquesten / parkview / hamilton beach / east industrial sector / normanhurst / homeside / east crown point)', 648: 'brantford central', 649: 'hamilton (confederation park / nashdale / east kentley / riverdale / lakely / grayside / north stoney creek)', 650: 'brantford southeast', 651: 'halton regional municipality (campbellville)', 652: 'frontenac county  addington county  loyalist shores and southwest leeds (inverary)', 653: 'brantford northwest', 654: 'hamilton (greenford / north gershome / west stoney creek)', 655: 'hamilton (west delta / blakeley / south stipley / south gibson / st. clair)', 656: 'hamilton (west industrial sector / west crown point / north stipley / north gibson / landsdale / keith / north end / beasley)', 657: "hamilton (east delta / bartonville / glenview / rosedale / lower king's forest / red hill / corman / vincent / south gershome)", 658: 'hamilton (east albion falls / south stoney creek)', 659: 'hamilton (stinson / corktown)', 660: 'hamilton (central / strathcona / south dundurn)', 661: 'hamilton (durand / kirkendall / chedoke park)', 663: 'hamilton (raleigh / macassa / lawfield / thorner / burkholme / eastmount)', 664: 'hamilton (crerar / bruleville / hill park / inch park / centremount / balfour / greeningdon / jerome)', 665: 'hamilton (barnstown / west chappel / allison / ryckmans / mewburn / sheldon / falkirk / carpenter / kennedy / southwest outskirts)', 666: 'hamilton (southam / bonnington / yeoville / kernighan / gourley / rolston / buchanan / mohawk / westcliffe / gilbert / gilkson / gurnett / fessenden / mountview)', 668: "hamilton (sherwood / huntington / upper king's forest / lisgar / berrisfield / hampton heights / sunninghill)", 669: 'renfrew county and lanark highlands township (deep river)', 670: 'hamilton (westdale / cootes paradise / ainslie wood)', 672: 'hamilton (west albion falls / hannon / rymal / trenholme / quinndale / templemead / broughton / eleanor / randall / rushdale / butler / east chappel)', 673: 'peterborough county and north hastings county (lakefield)', 710: 'ottawa (lower town / sandy hill / university of ottawa)', 711: 'government of canada ottawa and gatineau offices', 712: 'ottawa (alta vista)', 713: 'ottawa (riverview / hawthorne)', 715: 'ottawa (rockcliffe park / new edinburgh)', 716: 'ottawa (vanier)', 717: 'ottawa (parliament hill)', 719: 'ottawa (overbrook)', 720: 'ottawa (west downtown area)'}, 'adminName1': {735: 'ontario', 662: 'ontario', 667: 'ontario', 671: 'ontario', 674: 'ontario', 675: 'ontario', 676: 'ontario', 677: 'ontario', 678: 'ontario', 679: 'ontario', 680: 'ontario', 681: 'ontario', 682: 'ontario', 683: 'ontario', 684: 'ontario', 685: 'ontario', 686: 'ontario', 687: 'ontario', 688: 'ontario', 689: 'ontario', 690: 'ontario', 691: 'ontario', 692: 'ontario', 693: 'ontario', 694: 'ontario', 695: 'ontario', 696: 'ontario', 697: 'ontario', 698: 'ontario', 699: 'ontario', 700: 'ontario', 701: 'ontario', 702: 'ontario', 703: 'ontario', 704: 'ontario', 705: 'ontario', 706: 'ontario', 707: 'ontario', 708: 'ontario', 709: 'ontario', 714: 'ontario', 718: 'ontario', 641: 'ontario', 642: 'ontario', 643: 'ontario', 644: 'ontario', 645: 'ontario', 646: 'ontario', 648: 'ontario', 649: 'ontario', 650: 'ontario', 651: 'ontario', 652: 'ontario', 653: 'ontario', 654: 'ontario', 655: 'ontario', 656: 'ontario', 657: 'ontario', 658: 'ontario', 659: 'ontario', 660: 'ontario', 661: 'ontario', 663: 'ontario', 664: 'ontario', 665: 'ontario', 666: 'ontario', 668: 'ontario', 669: 'ontario', 670: 'ontario', 672: 'ontario', 673: 'ontario', 710: 'ontario', 711: 'ontario', 712: 'ontario', 713: 'ontario', 715: 'ontario', 716: 'ontario', 717: 'ontario', 719: 'ontario', 720: 'ontario'}, 'adminCode1': {735: 'ON', 662: 'ON', 667: 'ON', 671: 'ON', 674: 'ON', 675: 'ON', 676: 'ON', 677: 'ON', 678: 'ON', 679: 'ON', 680: 'ON', 681: 'ON', 682: 'ON', 683: 'ON', 684: 'ON', 685: 'ON', 686: 'ON', 687: 'ON', 688: 'ON', 689: 'ON', 690: 'ON', 691: 'ON', 692: 'ON', 693: 'ON', 694: 'ON', 695: 'ON', 696: 'ON', 697: 'ON', 698: 'ON', 699: 'ON', 700: 'ON', 701: 'ON', 702: 'ON', 703: 'ON', 704: 'ON', 705: 'ON', 706: 'ON', 707: 'ON', 708: 'ON', 709: 'ON', 714: 'ON', 718: 'ON', 641: 'ON', 642: 'ON', 643: 'ON', 644: 'ON', 645: 'ON', 646: 'ON', 648: 'ON', 649: 'ON', 650: 'ON', 651: 'ON', 652: 'ON', 653: 'ON', 654: 'ON', 655: 'ON', 656: 'ON', 657: 'ON', 658: 'ON', 659: 'ON', 660: 'ON', 661: 'ON', 663: 'ON', 664: 'ON', 665: 'ON', 666: 'ON', 668: 'ON', 669: 'ON', 670: 'ON', 672: 'ON', 673: 'ON', 710: 'ON', 711: 'ON', 712: 'ON', 713: 'ON', 715: 'ON', 716: 'ON', 717: 'ON', 719: 'ON', 720: 'ON'}, 'adminName2': {735: "dsad", 662: 'toronto', 667: 'toronto', 671: 'toronto', 674: 'toronto', 675: 'toronto', 676: 'toronto', 677: 'toronto', 678: 'toronto', 679: 'toronto', 680: 'toronto', 681: 'toronto', 682: 'toronto', 683: 'toronto', 684: 'toronto', 685: 'toronto', 686: 'toronto', 687: 'toronto', 688: 'toronto', 689: 'toronto', 690: 'toronto', 691: 'toronto', 692: 'toronto', 693: 'toronto', 694: 'toronto', 695: 'toronto', 696: 'toronto', 697: 'toronto', 698: 'toronto', 699: 'toronto', 700: 'toronto', 701: 'toronto', 702: 'toronto', 703: 'toronto', 704: 'toronto', 705: 'toronto', 706: 'toronto', 707: 'toronto', 708: 'toronto', 709: 'toronto', 714: 'toronto', 718: 'toronto', 641: 'dufferin county', 642: 'algoma', 643: 'halton', 644: 'brant', 645: 'brant', 646: 'hamilton', 648: 'brant', 649: 'hamilton', 650: 'brant', 651: 'halton', 652: 'frontenac county', 653: 'brant', 654: 'hamilton', 655: 'hamilton', 656: 'hamilton', 657: 'hamilton', 658: 'hamilton', 659: 'hamilton', 660: 'hamilton', 661: 'hamilton', 663: 'hamilton', 664: 'hamilton', 665: 'hamilton', 666: 'hamilton', 668: 'hamilton', 669: 'renfrew county', 670: 'hamilton', 672: 'hamilton', 673: 'peterborough county', 710: 'ottawa', 711: 'ottawa', 712: 'ottawa', 713: 'ottawa', 715: 'ottawa', 716: 'ottawa', 717: 'ottawa', 719: 'ottawa', 720: 'ottawa'}, 'adminCode2': {735: '', 662: '8133394', 667: '8133394', 671: '8133394', 674: '8133394', 675: '8133394', 676: '8133394', 677: '8133394', 678: '8133394', 679: '8133394', 680: '8133394', 681: '8133394', 682: '8133394', 683: '8133394', 684: '8133394', 685: '8133394', 686: '8133394', 687: '8133394', 688: '8133394', 689: '8133394', 690: '8133394', 691: '8133394', 692: '8133394', 693: '8133394', 694: '8133394', 695: '8133394', 696: '8133394', 697: '8133394', 698: '8133394', 699: '8133394', 700: '8133394', 701: '8133394', 702: '8133394', 703: '8133394', 704: '8133394', 705: '8133394', 706: '8133394', 707: '8133394', 708: '8133394', 709: '8133394', 714: '8133394', 718: '8133394', 641: '5943542', 642: '5883638', 643: '5969719', 644: '5907982', 645: '5907982', 646: '5969784', 648: '5907982', 649: '5969784', 650: '5907982', 651: '5969719', 652: '5958506', 653: '5907982', 654: '5969784', 655: '5969784', 656: '5969784', 657: '5969784', 658: '5969784', 659: '5969784', 660: '5969784', 661: '5969784', 663: '5969784', 664: '5969784', 665: '5969784', 666: '5969784', 668: '5969784', 669: '6119449', 670: '5969784', 672: '5969784', 673: '6101646', 710: '8581623', 711: '8581623', 712: '8581623', 713: '8581623', 715: '8581623', 716: '8581623', 717: '8581623', 719: '8581623', 720: '8581623'}})

if __name__ == "__main__":
    pass

data = {"m3k", "downsview east (cfb toronto)", "toronto", "ontario"}








# def normalize(text):
#     return re.sub(r'\W+', ' ', text.lower()).strip()

# # Tokenize function
# def tokenize(text):
#     return [token.strip() for token in re.split(r'\s+|[/()|]', text) if token.strip()]

# user_input = "toronto"
# normalized_input = normalize(user_input)
# tokens_input = tokenize(normalized_input)

# df1 = df['placeName'].to_list(), lambda x : normalize(tokenize(x))

# def calculate_similarity_score(input_tokens, dictionary_entry):
#     tokens_dict = tokenize(normalize(dictionary_entry))
#     total_distance = sum(min(levenshtein_distance(token_input, token_dict) for token_dict in tokens_dict) for token_input in input_tokens)
#     return total_distance

# # Find the best match
# best_match = min(df1, key=lambda entry: calculate_similarity_score(tokens_input, entry))
# best_match_score = calculate_similarity_score(tokens_input, best_match)
























# path = "C:\\Users\\Jernej\\Documents\\OOP\\database_files\\naslovi1.csv"

# def clean_naslovi_csv():
#     data = pd.DataFrame(pd.read_csv(path)).drop(columns= ['Zaporedno številko vključitve','Angleška kategorije in vključitve']).rename(columns= {"Kategorije in vključitve" : "Naslov", "Šifra Kategorije" : "Sifra"})

#     data["Naslov"] = data["Naslov"].str.lower()

#     correction_dict = {
#         '?olska ulica': 'Čolska ulica',
#         '?rnomaljska cesta': 'Črnomaljska cesta',
#         '?tefanov trg': 'Štefanov trg',
#         'k taj?birtu': 'k tajčbirtu',
#         'kr?': 'krč',
#         'metli?ka cesta': 'metliška cesta',
#         'pod primo?em': 'pod primožem',
#         'pri po?ti': 'pri pošti',
#         'ro?ka cesta': 'roška cesta',
#         'smu?ka cesta': 'smučka cesta',
#         'spodnja ka??a': 'spodnja kašča',
#         'vavp?a vas': 'vavpča vas',
#         'vrta?a': 'vrtača',
#         'zgornja ka??a': 'zgornja kašča'
#     }

#     # Function to correct address
#     def correct_address(address):
#         return correction_dict.get(address, address)

#     data['Naslov'] = data['Naslov'].apply(correct_address)


# class Get_Slo_Naslovi:
#     def __init__(self, input) -> None:
#         self.input = input
#         self.path = "C:\\Users\\Jernej\\Documents\\OOP\\database_files\\naslovi1.csv"

#     async def read_csv(self, path):
#         loop = asyncio.get_event_loop()
#         data = await loop.run_in_executor(None, pd.read_csv, path)
#         values = list(data["Naslov"])
#         return values

#     async def get_naslovi_data(self, path):
#         data_task = asyncio.create_task(self.read_csv(path))
#         data = await data_task
#         matcher = FindAMatch(value= self.input, choices=data).find_match()
#         return matcher
    
#     async def run(self):
#         result = await asyncio.gather(self.get_naslovi_data(path))
#         return result

# if __name__ == '__main__':

#     start = time.perf_counter()
#     data = asyncio.run(Get_Slo_Naslovi("Vransko").run())
#     end = time.perf_counter()

#     print(data)
#     print(end - start)
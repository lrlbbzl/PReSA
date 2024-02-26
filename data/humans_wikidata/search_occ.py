# import requests
# import json
# from tqdm import tqdm
# import urllib
# import sys
# import http.client

# from wikidata.client import Client

# # 创建Wikidata客户端

# _entities = json.load(open('entities.json', 'r', encoding='utf-8'))
# entities = [x['entity_id'] for x in _entities]

# start = 24652
# entity = entities[start:]

# mp = json.load(open('occupations.json', 'r'))
# length = len(entity)

# i = 0

# while i < length:
#     try:
#         client = Client()
#         ent = entities[i]
#         # 获取实体ID为Q40的实体
#         entity = client.get(ent, load=True)

#         # 获取实体的所有属性声明
#         x = entity.__dict__
#         data = x['data']

#         claims = data['claims']
#         if 'P106' in claims:
#             occ = claims['P106']
#             occupations = [temp['mainsnak']['datavalue']['value']['id'] for temp in occ]
#             mp.update({ent : occupations})
#         if (i + start) % 10 == 0:
#             print(i + start)
#         i += 1

#     except http.client.RemoteDisconnected:
#         continue
#     except urllib.error.URLError:
#         continue
#     except KeyboardInterrupt:
#         print(i + start)
#         json.dump(mp, open('occupations.json', 'w'))
#         break
# json.dump(mp, open('occupations.json', 'w'))



import json
import pickle
import requests
import json
from tqdm import tqdm
import urllib
import sys
import http.client

from wikidata.client import Client
cand = pickle.load(open('candidates.pkl', 'rb'))

length = len(cand)
i = 0
mp = dict()

while i < length:
    try:
        client = Client()
        ent = cand[i]
        # 获取实体ID为Q40的实体
        entity = client.get(ent, load=True)

        if entity.label:
            mp.update({ent : entity.label})
        
        if i % 20 == 0:
            print(i)
        i += 1

    except http.client.RemoteDisconnected:
        continue
    except urllib.error.URLError:
        continue
    except TypeError:
        i += 1
        continue
    except KeyboardInterrupt:
        print(i)
        json.dump(mp, open('occ2name.json', 'w'))
        break
pickle.dump(mp, open('occ2name.pkl', 'wb'))

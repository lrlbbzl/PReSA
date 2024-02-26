from wikidata.client import Client
import json
import urllib
import requests
import sys
import pickle
import http.client

def merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if k in a:
            a[k] = a[k] | v
        else:
            a.update({k : v})
    return a


entity2type = pickle.load(open('entity2type.pkl', 'rb'))
entities = pickle.load(open('entity.pkl', 'rb'))
p = open('null.txt', 'r')
null_entities = [line.strip() for line in p.readlines()]
res = pickle.load(open('null_data_types.pkl', 'rb'))

i = 5382
gap = 50
length = len(null_entities)
while i < length:
    try:
        entity = null_entities[i]
        if entity not in res:
            res.update({entity : set()})
        client = Client()
        x = client.get(entity, load=True)
        claims = x.data['claims']
        if 'P31' in claims:
            instances = claims['P31']
            for j in range(len(instances)):
                res[entity].add(instances[j]['mainsnak']['datavalue']['value']['id'])
        elif 'P279' in claims:
            instances = claims['P279']
            for j in range(len(instances)):
                res[entity].add(instances[j]['mainsnak']['datavalue']['value']['id'])
        i += 1
        if i % gap == 0:
            print(i)
            sys.stdout.flush()
    except urllib.error.HTTPError as err:
        i += 1
        if i % gap == 0:
            print(i)
            sys.stdout.flush()
        continue
    except http.client.RemoteDisconnected:
        pickle.dump(res, open('null_data_types.pkl', 'wb'))
        continue
    except urllib.error.URLError:
        pickle.dump(res, open('null_data_types.pkl', 'wb'))
        continue
    except http.client.IncompleteRead:
        i += 1
        if i % gap == 0:
            print(i)
            sys.stdout.flush()
        continue
    except ValueError:
        i += 1
        if i % gap == 0:
            print(i)
            sys.stdout.flush()
        continue
    except KeyboardInterrupt:
        pickle.dump(res, open('null_data_types.pkl', 'wb'))
        break

pickle.dump(res, open('null_data_types.pkl', 'wb'))
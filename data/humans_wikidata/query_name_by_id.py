from wikidata.client import Client
import json
import urllib
import requests
import sys
import http.client
import pickle

x = open('null_data_type.txt', 'r')

start = 0

lines = x.readlines()
lines = lines[3801:]
no = []

res = dict()
i = 0
gap = 1

while i < len(lines):
    try:
        entity_id, idx = lines[i].strip().split('\t')
        client = Client()
        entity = client.get(idx, load=True)
        if entity_id not in res:
            res.update({entity_id : set()})
        if entity is None:
            no.append(entity_id)
        # 解析查询结果
        elif entity.label == {}:
            no.append(entity_id)
        else:
            name = entity.label.get('en', "")
            res[entity_id].add(name)
        i += 1
        if i % gap == 0:
            print(i + start)
            sys.stdout.flush()
    except urllib.error.HTTPError as err:
        i += 1
        if i % gap == 0:
            print(i + start)
            sys.stdout.flush()
        continue
    except http.client.RemoteDisconnected:
        pickle.dump(res, open('null_entity2type.pkl', 'wb'))
        continue
    except urllib.error.URLError:
        pickle.dump(res, open('null_entity2type.pkl', 'wb'))
        continue
    except http.client.IncompleteRead:
        i += 1
        if i % gap == 0:
            print(i + start)
            sys.stdout.flush()
        continue
    except ValueError:
        i += 1
        if i % gap == 0:
            print(i + start)
            sys.stdout.flush()
        continue
    except KeyboardInterrupt:
        pickle.dump(res, open('null_entity2type.pkl', 'wb'))
        break

pickle.dump(res, open('null_entity2type.pkl', 'wb'))
print('\n'.join(no))
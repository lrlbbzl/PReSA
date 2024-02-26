import pickle

import json
mp = json.load(open('relations.json', 'r'))

for fil in ['train.txt.json', 'test.txt.json', 'valid.txt.json']:
    p = json.load(open(fil, 'r'))
    for i in range(len(p)):
        p[i]['relation'] = mp[p[i]['relation']]
    json.dump(p, open(fil, 'w'))
    
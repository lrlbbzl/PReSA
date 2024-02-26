import pickle

entity = pickle.load(open('entity.pkl', 'rb'))

def deal(file):
    x = open(file, 'r')
    res = []
    for line in x.readlines():
        a, b, c = line.strip().split('\t')
        if a not in entity or c not in entity:
            continue
        else:
            res.append(a + '\t' + b + '\t' + c + '\n')
    p = open('new_' + file, 'w')
    p.writelines(res)

for l in ['train.txt', 'valid.txt', 'test.txt']:
    deal(l)
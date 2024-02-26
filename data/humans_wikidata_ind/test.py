import pickle
train, valid, test = pickle.load(open('trains.pkl', 'rb')), pickle.load(open('valids.pkl', 'rb')), pickle.load(open('tests.pkl', 'rb'))

x = open('train.txt', 'w')
x.writelines(train)

x = open('valid.txt', 'w')
x.writelines(valid)

x = open('test.txt', 'w')
x.writelines(test)
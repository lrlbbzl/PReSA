import requests

from wikidata.client import Client

# 创建Wikidata客户端
client = Client()

# 获取实体ID为Q40的实体
entity = client.get("Q40", load=True)

# 获取实体的所有属性声明
x = entity.__dict__
data = x['data']

claims = data['claims']
instance_of = claims['P31']
# subclass_of = claims['P279']
print(len(instance_of))
print(instance_of[-1]['mainsnak']['datavalue']['value']['id'])
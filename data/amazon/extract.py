import json
import os
from collections import namedtuple

d = 'D:\\New folder\\Data\\alexa-prize-topical-chat-dataset-master\\conversations'
s = './data/amazon'

for f in os.listdir(d):
    data = open(os.path.join(d, f),'r').read()
    data = json.loads(data)
    with open(os.path.join(s, f), 'w') as w:
        json.dump(data, w)





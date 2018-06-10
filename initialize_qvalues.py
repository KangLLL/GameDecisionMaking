import json
from model_free import Model_free

# Script to create Q-Value JSON file, initilazing with zeros
qval = {"420_240_0":[0, 0]}
# qval = {"420_240_0_0_0":[0, 0]}
model_free = Model_free()
fd = open(model_free.addr + '/qvalues.json', 'w')
json.dump(qval, fd)
fd.close()

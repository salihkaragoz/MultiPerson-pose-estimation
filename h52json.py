import h5py
import json
import sys

# read hdf5 file
h5_file = sys.argv[1]
f = h5py.File(h5_file, 'r')
key = f.keys()[0]
data = list(f[key])
data = data[:-1]

# read text file
txt_file = sys.argv[2]
with open(txt_file, 'r') as f:
    X = [tuple(i.split(' ')) for i in f]
    X = zip(*X)
    names = X[0]
    ids = map(lambda x: int(x[:-4]), names)
    scores = map(float, X[1])
    det = zip(ids,scores)
keypoints = zip(det, data)
# if test-dev exclude test-challenge
if sys.argv[3] == 'dev':
    print 'yes'
    f = open('image_info_test-dev2017.json')
    data = json.loads(f.read())
    idxs = []
    for d in data['images']:
    	idxs.append(int(d['id']))
    keypoints_dev = []
    for k in keypoints:
    	if k[0][0] in idxs:
    		keypoints_dev.append(k)
    keypoints = keypoints_dev
    
# combine them
result = []

for k in keypoints:
    ann = dict.fromkeys(['image_id', 'category_id', 'keypoints', 'score'])
    ann['image_id'] = k[0][0]
    ann['score'] = k[0][1]
    ann['category_id'] = 1
    ks = []
    for i in k[1]:
        ks.append(i[0])
        ks.append(i[1])
        ks.append(1)
    ann['keypoints'] = ks
    result.append(ann)
# save json
with open('result_%s.json'%txt_file[10:-3], 'w') as f:
    json.dump(result, f)

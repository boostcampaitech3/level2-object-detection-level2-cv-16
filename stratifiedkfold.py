import json 
import numpy as np 
import funcy
import argparse
from sklearn.model_selection import StratifiedGroupKFold 

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('--annotations', default='/opt/ml/detection/dataset/train.json', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('--train', default='/opt/ml/detection/dataset/kfold', type=str, help='Where to store COCO training annotations')
parser.add_argument('--val', default='/opt/ml/detection/dataset/kfold', type=str, help='Where to store COCO test annotations')
parser.add_argument('--splits', type=int, default = 5)

args = parser.parse_args()

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

with open(args.annotations) as f: 
    coco = json.load(f)
    info = coco['info']
    licenses = coco['licenses']
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

var = [(ann['image_id'], ann['category_id']) for ann in annotations]
X = np.ones((len(annotations),1))
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var]) 

cv = StratifiedGroupKFold(n_splits=args.splits, shuffle=True, random_state=411) 

for i,(train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
    train_idx =np.unique(groups[train_idx])
    val_idx = np.unique(groups[val_idx])
    save_coco(args.train+'/train' + str(i)+'.json', info, licenses, [images[i] for i in train_idx], filter_annotations(annotations, [images[i] for i in train_idx]), categories)
    save_coco(args.val+'/val' + str(i)+'.json', info, licenses, [images[i] for i in val_idx], filter_annotations(annotations, [images[i] for i in val_idx]), categories)
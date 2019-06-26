import pickle
import os
import utill as ut
from pathlib import Path

train_folder = "./dataset/images_background/"
val_folder = './dataset/images_evaluation/'
save_path = Path('./save/')
save_path.mkdir(parents=True, exist_ok=True)

X,y,c = ut.loadimgs(train_folder)
Xval,yval,cval = ut.loadimgs(val_folder)

with open(os.path.join(save_path,"train.pickle"), "wb") as f:
    pickle.dump((X, c), f)
    
with open(os.path.join(save_path,"val.pickle"), "wb") as f:
    pickle.dump((Xval, cval), f)
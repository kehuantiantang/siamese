import pickle
import os
import time
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from tqdm import tqdm, trange

import utill as ut
from model import base as md
from pathlib import Path

save_path = Path('./save/')
model_path = Path('./weights/')
save_path.mkdir(parents=True, exist_ok=True)
model_path.mkdir(parents=True, exist_ok=True)

with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)

with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)

model = md.get_model((105, 105, 1), backbone=None, pre_train=True)
gpu_model = multi_gpu_model(model, gpus=4)
gpu_model.summary()
optimizer = Adam(lr=0.00006)
gpu_model.compile(loss="binary_crossentropy", optimizer=optimizer)

evaluate_every = 50  # interval for evaluating on one-shot tasks
batch_size = 128
epoch = 10000

n_iter = 20000  # No. of training iterations
N_way = 20  # how many classes for testing one-shot tasks
n_val = 250  # how many one-shot tasks to validate on
best = -1

print("Starting training process!")
print("-------------------------------------")
t_start = time.time()

bar = trange(epoch, desc='1st loop')

loss = 1
for i in bar:
    for j in range(len(Xtrain) // batch_size):
        (inputs, targets) = ut.get_batch(batch_size, Xtrain, train_classes)
        loss = gpu_model.train_on_batch(inputs, targets)

        bar.set_description(
            'Loss: %.5f, Time: %f' % (loss, (time.time(

            ) - t_start) / 60.0))

    bar.set_description("Train Loss: {0}".format(loss))

    if i % evaluate_every == 0:
        val_acc = ut.test_oneshot(gpu_model, N_way, n_val, Xval, val_classes,
                                  verbose=True)
        if val_acc >= best:
            print(
                "\nCurrent best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc

            gpu_model.save(os.path.join(model_path, 'weights_best.h5'.format(i)))

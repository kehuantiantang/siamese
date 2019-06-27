import pickle
import os
import time
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model
from tqdm import tqdm, trange

import utill as ut
from model import base as md
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

save_path = Path('./save/')
model_path = Path('./weights/')
save_path.mkdir(parents=True, exist_ok=True)
model_path.mkdir(parents=True, exist_ok=True)

with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)

with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)

evaluate_every = 50  # interval for evaluating on one-shot tasks
batch_size = 128
epoch = 3000

N_way = 20  # how many classes for testing one-shot tasks
n_val = 250  # how many one-shot tasks to validate on
best = -1

loss = -1
lr = 6e-4
input_shape = (105, 105, 1)
backbone = None
is_train = True
is_test = True

checkpoint_path = None  # '/home/khtt/code/siamese/weights/resnet50_best.h5'

model = md.get_model(input_shape, backbone=backbone, pre_train=True)
gpu_model = multi_gpu_model(model, gpus=3)
gpu_model.name = model.name
# gpu_model.summary()
optimizer = SGD(lr=lr, momentum=0.9, nesterov=True)  # 0.00006
gpu_model.compile(loss="binary_crossentropy", optimizer=optimizer)

if checkpoint_path:
    gpu_model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
    print(f'Load weight from {checkpoint_path}')

print("Starting training process!")
print("-------------------------------------")
t_start = time.time()

if is_train:
    bar = trange(epoch)
    for i in bar:
        for j in range(len(Xtrain) // batch_size):
            (inputs, targets) = ut.get_batch(batch_size, Xtrain, train_classes)
            loss = gpu_model.train_on_batch(inputs, targets)

            bar.set_description('Loss: %.5f, Time: %f' % (loss, (time.time(

            ) - t_start) / 60.0))

        if i % evaluate_every == 0:
            val_acc = ut.test_oneshot(gpu_model, N_way, n_val, Xval,
                                      val_classes, verbose=True)
            if val_acc >= best:
                print(
                    "\n ------------------ Current best: {0}, previous best: {1}".format(
                        val_acc, best))
                best = val_acc

                gpu_model.save(os.path.join(model_path,
                                            '{}_best_backbone.h5'.format(
                    gpu_model.name)))

    gpu_model.save(
        os.path.join(model_path, '{}_final1_backbone.h5'.format(gpu_model.name)))

if is_train:
    val_acc = ut.test_oneshot(gpu_model, N_way, n_val, Xval, val_classes,
                              verbose=True)
    print("\n ------------------ Current best: {0}, previous best: {1}".format(
        val_acc, best))
    best = val_acc

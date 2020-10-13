import tensorflow as tf, numpy as np, os, sys
print(tf.__version__)
from cnn_sys_ident.architectures.models import BaseModel, CorePlusReadoutModel
from cnn_sys_ident.architectures.cores import StackedRotEquiHermiteConv2dCore
from cnn_sys_ident.architectures.readouts import SpatialXFeatureJointL1Readout
from cnn_sys_ident.architectures.training import Trainer
from analysis.iclr2019.data import Dataset

DATA_FILE = 'data.pkl'
NUM_ROTATIONS = 8
UPSAMPLING = 2
SHARED_BIASES = False
FILTER_SIZE = [13, 5, 5]
NUM_FILTERS = [16, 16, 16]
STRIDE = [1, 1, 1]
RATE = [1, 1, 1]
PADDING = ['SAME', 'SAME', 'SAME']
ACTIVATION_FN = ['soft', 'soft', 'none']
REL_SMOOTH_WEIGHT = [1, 0.5, 0.5]
REL_SPARSE_WEIGHT = [0, 1, 1]

# Readout
POSITIVE_FEATURE_WEIGHTS = False
INIT_MASKS = 'rand'

# Training
VAL_STEPS = 50
LEARNING_RATE = 0.002
BATCH_SIZE = 256
PATIENCE = 5
LR_DECAY_STEPS = 2
LOG_DIR = 'analysis/iclr2019/checkpoints-repro-new'

data = Dataset.load(DATA_FILE)

base = BaseModel(
    data
)
core = StackedRotEquiHermiteConv2dCore(
    base,
    base.inputs,
    num_rotations=NUM_ROTATIONS,
    upsampling=UPSAMPLING,
    shared_biases=SHARED_BIASES,
    filter_size=FILTER_SIZE,
    num_filters=NUM_FILTERS,
    stride=STRIDE,
    rate=RATE,
    padding=PADDING,
    activation_fn=ACTIVATION_FN,
)
readout = SpatialXFeatureJointL1Readout(
    base,
    core.output,
    positive_feature_weights=POSITIVE_FEATURE_WEIGHTS,
)
model = CorePlusReadoutModel(base, core, readout)
#model.load()
trainer = Trainer(base, model)   # just for computing the performance
trainer.fit()
trainer.compute_test_corr()      #   (for training, see below)



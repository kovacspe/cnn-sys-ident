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

log_hash = {   # determines the seed of the random number generator
    8:  '8d2912ce0669f4dcc4efa78b970e453c',
    12: '4d2e43901a1be496a5e66dc9fec1ed14',
    16: '647bb1d1bd02979996e492b5422eb95f',
    20: '6babf3b3be2cbd8da50e091966f22e46',
    24: '1e34d6f792b506630897ce84fe93a58c',
    28: 'a653720bdd962f95b213156f25c80f31',
    32: 'd23dd9d3a7149ecc72627115bb940e1e',
    40: 'ba65e73469fe90109f22e8204557b646',
    48: '37e70606daaa0b2ca13698fee329eec4'}

conv_smooth_weight = {
    8:  0.00781004, 12: 0.00184694, 16: 0.0249692,
    20: 0.0257738,  24: 0.00146371, 28: 0.0186784,
    32: 0.026082,   40: 0.00232312, 48: 0.00129107}
conv_sparse_weight = {
    8:  0.0168574,  12: 0.0610123,  16: 0.0152482,
    20: 0.0691215,  24: 0.00999698, 28: 0.0187448,
    32: 0.0118641,  40: 0.0868334,  48: 0.0644271}
readout_sparsity = {
    8:  0.0156452,  12: 0.0153464,  16: 0.0170696,
    20: 0.0141163,  24: 0.0131784,  28: 0.0124147,
    32: 0.0161513,  40: 0.0115895,  48: 0.0163213}


data = Dataset.load(DATA_FILE)

num_features = 16
base = BaseModel(
    data,
    log_dir=LOG_DIR,
    log_hash=log_hash[num_features]
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
    rel_smooth_weight=REL_SMOOTH_WEIGHT,
    rel_sparse_weight=REL_SPARSE_WEIGHT,
    conv_smooth_weight=conv_smooth_weight[num_features],
    conv_sparse_weight=conv_sparse_weight[num_features],
)
readout = SpatialXFeatureJointL1Readout(
    base,
    core.output,
    positive_feature_weights=POSITIVE_FEATURE_WEIGHTS,
    init_masks=INIT_MASKS,
    readout_sparsity=readout_sparsity[num_features],
)
model = CorePlusReadoutModel(base, core, readout)
trainer = Trainer(base, model)
iter_num, val_loss, test_corr = trainer.fit(
    val_steps=VAL_STEPS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    patience=PATIENCE,
    lr_decay_steps=LR_DECAY_STEPS)

print('corr: ',trainer.compute_test_corr())
from yacs.config import CfgNode as CN

_C = CN()
_C.batch_size = 128
_C.num_epoch = 300
_C.dataset = 'cifar10'
_C.num_classes = 10
_C.crop_size = 224 # imagenet
_C.crop_interpolation = 2 # 2=BILINEAR, default; 3=BICUBIC
_C.optimizer = 'SGD'
_C.gamma = 0.1 # decay_rate
_C.use_cosine_lr = False
_C.cosine_lr_min = 0.0
_C.warmup_epochs = 5
_C.lr = 0.1
_C.lr_scale_factor = 256 # per nvidia apex
_C.lr_milestones = [150, 225]
_C.momentum = 0.9
_C.wd = 5e-4
_C.nesterov = False
_C.activation_mode = 0 # 1: leakyReLU, 2: ReLU6 , other: ReLU
_C.init_mode = 'kaiming'
_C.norm_name = 'BatchNorm2d'
_C.norm_groups = 0
_C.norm_k = [0]
_C.norm_attention_mode = 0
_C.norm_zero_gamma_init = False
_C.norm_all_mix = False

# data augmentation
_C.dataaug = CN()
_C.dataaug.imagenet_extra_aug = False
_C.dataaug.labelsmoothing_rate = 0. # 0.1
_C.dataaug.mixup_rate = 0.  # 0.2

# stem
_C.stem = CN()
_C.stem.imagenet_head7x7 = False
_C.stem.replace_maxpool_with_res_bottleneck = False
_C.stem.stem_kernel_size = 7
_C.stem.stem_stride = 2


# aognet
_C.aognet = CN()
_C.aognet.filter_list = [16, 64, 128, 256]
_C.aognet.out_channels = [0,0]
_C.aognet.blocks = [1, 1, 1]
_C.aognet.dims = [4, 4, 4]
_C.aognet.max_split = [2, 2, 2] # must >= 2
_C.aognet.extra_node_hierarchy = [0, 0, 0]  # 0: none, 1: tnode topdown, 2: tnode bottomup layerwise, 3: tnode bottomup sequential, 4: non-term node lateral, 5: tnode bottomup
_C.aognet.remove_symmetric_children_of_or_node = [0, 0, 0]
_C.aognet.terminal_node_no_slice = [0, 0, 0]
_C.aognet.stride = [1, 2, 2]
_C.aognet.drop_rate = [0.0, 0.0, 0.0]
_C.aognet.bottleneck_ratio = 0.25
_C.aognet.handle_dbl_cnt = True
_C.aognet.handle_tnode_dbl_cnt = False
_C.aognet.handle_dbl_cnt_in_param_init = False
_C.aognet.use_group_conv = False
_C.aognet.width_per_group = 0
_C.aognet.when_downsample = 0 # 0: at T-nodes, 1: before a aogblock, by conv_norm_ac + avgpool
_C.aognet.replace_stride_with_avgpool = True # for downsample in node op.
_C.aognet.use_elem_max_for_ORNodes = False

cfg = _C

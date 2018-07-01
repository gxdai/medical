#!/bin/bash

#!/bin/bash

# target: 0.8


CUDA_VISIBLE_DEVICES=$1 py_gxdai vgg16.py \
	--num_epochs1 200 \
	--batch_size 32 \
	--restore_ckpt 1 \
	--evaluation 1 \
	--ckpt_dir "./models/alchemic" \
	--dn_train 20 \
	--dn_test 1 \
	--weightFile "./models/old/my_model.ckpt-57"
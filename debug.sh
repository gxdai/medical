
#!/bin/bash

# target: 0.8


CUDA_VISIBLE_DEVICES=$1 py_gxdai debug.py \
	--num_epochs1 200 \
	--batch_size 128 \
	--restore_ckpt 0 \
	--evaluation 0 \
	--ckpt_dir "./models/alchemic/momentumOptimizer" \
	--dn_train 1 \
	--dn_test 1 \
	--weightFile "./models/alchemic/models-27" \
	--targetNum 110000

# --weightFile "./models/old/my_model.ckpt-57" \

  <h1> Quantum-Inspired Fusion for Open-domain Question Answering </h1>



# Development
## Setup
```
pip install -r requirements.txt
```
## Train Script
```
export DATASETS=NQ
export NGPU=4
export MODEL_SIZE=base
export QF=none
export N_CONTEXT=100
export NAME=${DATASETS}_${MODEL_SIZE}_QF=${QF}_NC=${N_CONTEXT}
# per_gpu_train_batch_size
export TBS=2
export EBS=$(($TBS * 2))
export AS=$((64 / $NGPU / $TBS))
export TS=$(($AS * 10000))
export EF=$((500 * $AS))

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=${NGPU} train_reader.py \
	        --train_data path\to\train\data \
	        --eval_data path\to\eval\data \
	        --model_size ${MODEL_SIZE} \
	        --per_gpu_train_batch_size $TBS \
	        --per_gpu_eval_batch_size $EBS \
	        --accumulation_steps $AS \
	        --total_steps $TS \
	        --eval_freq $EF \
	        --save_freq $EF \
	        --n_context $N_CONTEXT \
	        --add_loss binary \
	        --cat_emb \
                --name ${NAME} \
	        --checkpoint_dir checkpoint \
	        --use_checkpoint \
                --add_fusion qf_ds \
```
## Test Script
```
name={name}
CUDA_VISIBLE_DEVICES=0 python test_reader.py \
        --eval_data path\to\data \
        --model_path  path\to\model \
        --per_gpu_eval_batch_size 2 \
        --n_context 100 \
        --write_results \
        --answer_maxlength 30 \
        --name ${name} \
        --add_loss binary \
        --cat_emb \
        --output_attentions \
        --sum_golden_cross_att \
        --checkpoint_dir checkpoint \
        --add_fusion qf_ds
```

# Data
Following [FiD](https://github.com/facebookresearch/FiD), we use the same data and format.

You can also get the data with the following script.
```
sh get-data.sh
```


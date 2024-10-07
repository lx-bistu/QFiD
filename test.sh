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

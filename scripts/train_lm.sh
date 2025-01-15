# k_shots=(5)
k_shots=(5 10 20 50)

retrieval_stage1_modes=('task_specific_retrieval')
#retrieval_stage1_modes=('random' 'simcse_retrieval' 'task_specific_retrieval')

for k_shot in "${k_shots[@]}"
do

  CUDA_VISIBLE_DEVICES=3 python train_lm.py  --dataset TACRED  --k_shot $k_shot

done
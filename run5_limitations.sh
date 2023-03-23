# # MIND -> Adressa
for few_shot in  _200_4shot
do
  for few_shot_method in 2
  do
    for sim in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
    for random_seed in 42 0 100 1000 2022
    do
        CUDA_VISIBLE_DEVICES=1 python execute_few_shot_new_align_newscls.py  --few_shot=$few_shot --few_shot_method=$few_shot_method --range=Model/engTonor  --loss_weight=0.2 --target_domain_sim=$sim  --news_cls_iter=1  --random_seed=$random_seed --loss_weight_align=1
    done
    done
  done
done

# # MIND -> Adressa
for few_shot in _500_10shot _1500_10shot _2500_10shot _3500_10shot _4500_10shot _5500_10shot
do
  for few_shot_method in 2
  do
    for random_seed in 42 0 2022
    do
#        CUDA_VISIBLE_DEVICES=1 python execute_few_shot_new_align_newscls.py  --few_shot=$few_shot --few_shot_method=$few_shot_method --range=Model/engTonor  --loss_weight=0.2 --target_domain_sim=0.6  --news_cls_iter=0  --random_seed=$random_seed

        CUDA_VISIBLE_DEVICES=1 python execute_few_shot_new_align_newscls.py  --few_shot=$few_shot --few_shot_method=$few_shot_method --range=Model/engTonor  --loss_weight=0.2 --target_domain_sim=0.6  --news_cls_iter=1  --random_seed=$random_seed --loss_weight_align=1
    done
  done
done







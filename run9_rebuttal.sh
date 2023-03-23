

# # MIND -> Adressa

for topn in 2 3 4 5
do
    for random_seed in 42 0 100 1000 2022
    do
 CUDA_VISIBLE_DEVICES=1 python execute_few_shot_new_align_newscls.py  --few_shot=_200_4shot --few_shot_method=2 --range=Model/engTonor  --loss_weight=0.2 --target_domain_sim=0.6  --news_cls_iter=1  --random_seed=$random_seed --loss_weight_align=1  --topn=$topn
 done
 done


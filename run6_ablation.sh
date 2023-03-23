# 4 * 5
#RM     w  1
#NA    w  --
#RM+NA  w  1
#
#RM  wo  run
#NA  wo --
#RM+NA  wo  run

# # MIND -> Adressa
for few_shot in  _200_4shot
do
  for few_shot_method in 2
  do
    for news_align in 0 1
    do
    for random_seed in 42 0 100 1000 2022
    do
        CUDA_VISIBLE_DEVICES=0 python execute_few_shot_new_align_newscls.py  --few_shot=$few_shot --few_shot_method=$few_shot_method --range=Model/engTonor  --loss_weight=0.2 --target_domain_sim=0  --news_cls_iter=$news_align  --random_seed=$random_seed --loss_weight_align=1
    done
    done
  done
done


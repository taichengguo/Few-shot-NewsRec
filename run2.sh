# # # MIND -> Adressa
# for few_shot in _200_2shot _200_4shot _200_6shot
# do
#   for few_shot_method in 2
#   do
#     for random_seed in 42
#     do
#         python execute_few_shot_new_align_newscls.py  --few_shot=$few_shot --few_shot_method=$few_shot_method --range=Model/engTonor  --loss_weight=0.2 --target_domain_sim=0.6  --news_cls_iter=0  --random_seed=$random_seed

#         python execute_few_shot_new_align_newscls.py  --few_shot=$few_shot --few_shot_method=$few_shot_method --range=Model/engTonor  --loss_weight=0.2 --target_domain_sim=0.6  --news_cls_iter=1  --random_seed=$random_seed --loss_weight_align=1
#     done
#   done
# done



# # MIND -> Adressa
#for few_shot in _0_0shot
#do
#  for few_shot_method in 2
#  do
#    for random_seed in 42 0 100 1000 2022
#    do
#        CUDA_VISIBLE_DEVICES=1 python execute_few_shot_new_align_newscls.py  --few_shot=$few_shot --few_shot_method=$few_shot_method --range=Model/engTonor  --loss_weight=0.2 --target_domain_sim=0.6  --news_cls_iter=0  --random_seed=$random_seed
#
#        CUDA_VISIBLE_DEVICES=1 python execute_few_shot_new_align_newscls.py  --few_shot=$few_shot --few_shot_method=$few_shot_method --range=Model/engTonor  --loss_weight=0.2 --target_domain_sim=0.6  --news_cls_iter=1  --random_seed=$random_seed --loss_weight_align=1
#    done
#  done
#done



# Adressa -> MIND
for few_shot in _0_0shot
do
  for few_shot_method in 2
  do
    for random_seed in 42 0 100 1000 2022
    do
      CUDA_VISIBLE_DEVICES=1 python execute_few_shot_new_align_newscls.py --few_shot=$few_shot  --few_shot_method=$few_shot_method --range=Model/data --target_domain_sim=0.5 --random_seed=$random_seed --loss_weight=0.2 --news_cls_iter=0

      CUDA_VISIBLE_DEVICES=1 python execute_few_shot_new_align_newscls.py --few_shot=$few_shot  --few_shot_method=$few_shot_method --range=Model/data --target_domain_sim=0.5 --random_seed=$random_seed --loss_weight=0.2  --news_cls_iter=1  --loss_weight_align=1
    done
  done
done



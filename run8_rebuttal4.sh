

 # Adressa -> MIND
for lossw in 0.5  0.75   1
do
  for alignw in 0  0.25  0.5  0.75  1
  do
    for random_seed in 42 100 2022
    do
      CUDA_VISIBLE_DEVICES=1 python execute_few_shot_new_align_newscls4.py --few_shot=_0_0shot  --few_shot_method=2 --range=Model/final_data --target_domain_sim=0.5 --random_seed=$random_seed --loss_weight=$lossw  --news_cls_iter=1  --loss_weight_align=$alignw
 done
 done
 done


#for lossw in  1
#do
#  for alignw in  1
#  do
#    for random_seed in 100 2022
#    do
#      CUDA_VISIBLE_DEVICES=1 python execute_few_shot_new_align_newscls4.py --few_shot=_200_4shot  --few_shot_method=2 --range=Model/final_data --target_domain_sim=0.5 --random_seed=$random_seed --loss_weight=$lossw  --news_cls_iter=1  --loss_weight_align=$alignw
# done
# done
# done


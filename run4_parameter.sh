


# Adressa -> MIND
for few_shot in _0_0shot _200_2shot _200_4shot
do
  for few_shot_method in 2
  do
    for loss_weigh in 0.1 0.5 0.8 1
    do
      for random_seed in 42 0 100 1000 2022
      do
      CUDA_VISIBLE_DEVICES=0 python execute_few_shot_new_align_newscls.py --few_shot=$few_shot  --few_shot_method=$few_shot_method --range=Model/data --target_domain_sim=0.5 --random_seed=$random_seed --loss_weight=$loss_weigh --news_cls_iter=0
      done
    done
  done
done
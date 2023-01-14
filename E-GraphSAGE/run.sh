datasets=('CES-CIC' 'Darknet' 'ToN-IoT')
use_contrastive_learning=(True False)
use_adversarial_perturb=(True False)
use_focal_loss=(True False)

for dataset in ${datasets[@]}
do
  for cons in ${use_contrastive_learning[@]}
  do
    for adv in ${use_adversarial_perturb[@]}
    do
      for focal in ${use_focal_loss[@]}
      do
        python fit_model.py --dataset ${dataset} --cons ${cons} --adv ${adv} --focal ${focal}
      done
    done
  done
done
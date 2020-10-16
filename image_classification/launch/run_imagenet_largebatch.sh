#!/usr/bin/env bash

function runexp {

datapath="/fs/cml-datasets/ImageNet/ILSVRC2012"


gpu=${1}
model=${2}
opt=${3}
beta1=${4}
beta2=${5}
lr=${6}
warmup_epochs=${7}
grad_acc=${8}
flags=${9}

other_cmd="${flags}"

others_print="$(echo -e "${flags}" | tr -d '[:space:]')"

expname="imagenet-${model}-${opt}-lr${lr}-betas${beta1}_${beta2}-we${warmup_epochs}-gacc${grad_acc}${others_print}-onlybnnowd-smoothing0.1-resume2"

if [ ${opt} = "adabound" ]; then
other_cmd="${other_cmd} --final_lr 0.1"
fi


cmd="
CUDA_VISIBLE_DEVICES=${gpu}
python train_imagenet.py ${datapath} --arch ${model}
--opt ${opt} --beta1 ${beta1} --beta2-max ${beta2}
--lr ${lr} ${other_cmd} --comet-tag laprop
--grad-acc-steps ${grad_acc} --warmup-epochs ${warmup_epochs} --label-smoothing 0.1
--workers 16 --verbose --seed 91874
"
#--resume chks/imagenet-resnet50-MAdam-lr0.04-betas0.9_0.999-we20-gacc86--weight-decay0.08--use-adamw--beta2-min0.5--batch-size381--grad-clip1--linear-decay-onlybnnowd-smoothing0.1-resume/chk_last.pth

#--resume chks/imagenet-resnet50-MAdam-lr0.04-betas0.9_0.999-we20-gacc86--weight-decay0.12--use-adamw--beta2-min0.5--batch-size381--grad-clip1--linear-decay-onlybnnowd-smoothing0.1-resume/chk_last.pth


debug=0

if [ ${debug} -eq 0 ]; then
cmd="${cmd} --expname ${expname}
> logs/${expname}.log 2>&1 &"
echo "logs/${expname}.log"
fi

eval ${cmd}

}

# runexp         gpu         model     opt       beta1   beta2     lr     warmup_epochs grad_acc   flags
#runexp    0,1,2,3,4,5,6,7   resnet50  MAdam       0.9    0.999    0.028284    10            22     " --weight-decay 0.1  --use-adamw --beta2-min 0.5 --batch-size 744 --grad-clip 1 --linear-decay"
#runexp    0,1,2,3,4,5,6,7   resnet50  MAdam       0.9     0.999 0.028284    10            22     " --weight-decay 0.005 --use-adamw --beta2-min 0.5 --batch-size 744"
runexp          0,1,2,3      resnet50  MAdam       0.9     0.999  0.035      20            86     " --weight-decay 0.12  --use-adamw --beta2-min 0.5 --batch-size 381 --grad-clip 1 --linear-decay "
#runexp    0,1,2,3  resnet50  MAdam       0.9     0.999  0.05657      40            172     " --weight-decay 0.1  --use-adamw --beta2-min 0.5 --batch-size 381 --grad-clip 1 --linear-decay "


#runexp    0,1,2,3,4,5   resnet50  MAdam       0.9     0.999  0.028284    10            29     " --weight-decay 0.1 --use-adamw --beta2-min 0.5 --batch-size 564"

#runexp    0,1,2,3   resnet50  MAdam       0.9     0.999  0.035     10            29     " --weight-decay 0.1  --use-adamw --beta2-min 0.5 --batch-size 564 --grad-clip 1"
#
#runexp    0,1,2,3   resnet50  MAdam       0.9     0.999  0.020     10            29     " --weight-decay 0.1  --use-adamw --beta2-min 0.5 --batch-size 564 --grad-clip 1"
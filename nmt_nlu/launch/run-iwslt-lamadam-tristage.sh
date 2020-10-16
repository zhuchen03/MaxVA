#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=madam                                          # sets the job name if not set from environment
#SBATCH --array=1-25                                             # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output slurm-logs/%x_%A_%a.log                                   # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error slurm-logs/%x_%A_%a.log                                    # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --time=72:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger                                     # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                                         # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32gb                                              # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=TIME_LIMIT,FAIL,ARRAY_TASKS                 # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE,

function runexp {
tokens=4096

gpu=${1}
optimizer=${2}
lr=${3} # 3e-4
mom=${4}
b2_min=${5} #0.98
b2_max=${6} #0.98
warmup=${7} # 2000
eps=${8}
scheduler=${9}
hs=${10}
ds=${11}
wd=${12}
seed=${13}
flags=${14}
#wd=0.0001

adam_betas="'(0.9, ${b2_max})'"
total_updates=60000

others_print="$(echo -e "${flags}" | tr -d '[:space:]')"

other_params="${flags}"

if [ ${optimizer} = "lamadam" ]; then
opt_commands="--optimizer lamadam  --momentum 0.9 --beta-min ${b2_min} --varscale-beta ${b2_max} --varscale-eps ${eps} "
opt_str="lapropw_adv-b2_${b2_min}_${b2_max}"

elif [ ${optimizer} = "madam" ]; then
beta2_range="'(${b2_min}, ${b2_max})'"
opt_commands="--optimizer madam --beta1 0.9 --beta2-range ${beta2_range} --adam-eps ${eps} "
opt_str="madam-b2_${b2_min}_${b2_max}"

fi

if [ ${scheduler} = "isqrt" ]; then
scheduler_str="--lr ${lr} --lr-scheduler inverse_sqrt --warmup-updates ${warmup}  --max-update ${total_updates}"
elif [ ${scheduler} = "poly" ]; then
scheduler_str="--lr ${lr} --lr-scheduler polynomial_decay --warmup-updates ${warmup}  --total-num-update ${total_updates} --max-update ${total_updates}"
elif [ ${scheduler} = "tristage" ]; then
scheduler_str="--lr ${lr} --lr-scheduler tri_stage --warmup-steps ${warmup} --hold-steps ${hs} --decay-steps ${ds} --init-lr-scale 1e-2 --final-lr-scale 1e-2  --max-update ${total_updates}"
fi

export expname=iwslt-${opt_str}-${scheduler}-hs${hs}-ds${ds}-lr${lr}-wd${wd}-warm${warmup}-eps${eps}${others_print}-seed${seed}_2

export CUDA_VISIBLE_DEVICES=${gpu}

bleu_args="'{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}'"

cmd="
python train.py
    data-bin/iwslt14.tokenized.de-en
    --save-dir chks/${expname}
    --log-interval 10
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed
    ${opt_commands} --clip-norm 0.0
    ${scheduler_str}
    --dropout 0.3 --weight-decay ${wd}
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1
    --max-tokens ${tokens}
    --eval-bleu
    --keep-last-epochs 1
    --eval-bleu-args ${bleu_args}
    --eval-bleu-detok moses
    --eval-bleu-remove-bpe
    --eval-bleu-print-samples
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
    ${other_params} --seed ${seed}
    --fp16
"

debug=0
if [ ${debug} -eq 0 ]; then
logpath="logs/${expname}.log"
cmd="${cmd}  --comet --api-key OVyeR5Xs0GGidqMc8g1s9K3Lx --comet-project 'iwslt-de-en-neurips' --comet-real-tag neurips  --comet-tag ${expname}
> ${logpath} 2>&1 &
"
echo ${logpath}
fi

#echo ${cmd}
eval ${cmd}

}
wd=1e-2

#source ~/cmlscratch/anaconda3/etc/profile.d/conda.sh
#conda activate base

#seed_list=(65443 9876 5454 3234 4543112)
#lr_list=(1e-3 1.25e-3 1.5e-3 1.75e-3 7.5e-4 )
#
#sidx=$(( (${SLURM_ARRAY_TASK_ID} - 1) % 5 ))
#lidx=$(( (${SLURM_ARRAY_TASK_ID} - 1) / 5 % 5 ))
#
#seed="${seed_list[sidx]}"
#lr="${lr_list[lidx]}"
#
## runexp gpu optimizer     lr    mom   b2_min      b2_max   warmup   eps   scheduler       hs     ds    wd   seed  flags
## please refer to hyper parameter settings in the appendix
##runexp    0   lamadam      1.5e-3   0.9    0.5       0.999     4000   1e-15    tristage        32000  24000  ${wd}   1234
#
#runexp    0   madam      ${lr}   0.9    0.5       0.999     4000   1e-7    tristage        32000  24000  ${wd}   ${seed}
#runexp    1   madam      1.25e-3   0.9    0.5       0.999     4000   1e-7    tristage        32000  24000  ${wd}   65443
runexp    0   madam      1.5e-3   0.9    0.5       0.999     4000   1e-7    tristage        32000  24000  ${wd}   65443
#runexp    2   madam      1.75e-3   0.9    0.5       0.999     4000   1e-7    tristage        32000  24000  ${wd}   65443

#runexp    2   madam      1.25e-3   0.9    0.5       0.999     4000   1e-7    tristage        32000  24000  ${wd}   9876
#runexp    3   madam      2e-3   0.9    0.5       0.999     4000   1e-7    tristage        32000  24000  ${wd}   5454


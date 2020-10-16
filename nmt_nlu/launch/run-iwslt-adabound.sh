#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=adabound                                          # sets the job name if not set from environment
#SBATCH --array=1-10                                             # Submit 8 array jobs, throttling to 4 at a time
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
lr=${2} # 3e-4
gamma=${3}
final_lr=${4}
warmup=${5} # 2000
eps=${6}
scheduler=${7}
hs=${8}
ds=${9}
wd=${10}
seed=${11}
flags=${12}

#adam_betas="'(0.9, ${b2_max})'"
total_updates=60000

others_print="$(echo -e "${flags}" | tr -d '[:space:]')"

other_params="${flags}"

opt_str=adabound_gamma${gamma}_flr${final_lr}

if [ ${scheduler} = "isqrt" ]; then
scheduler_str="--lr ${lr} --lr-scheduler inverse_sqrt --warmup-updates ${warmup}  --max-update ${total_updates}"
elif [ ${scheduler} = "poly" ]; then
scheduler_str="--lr ${lr} --lr-scheduler polynomial_decay --warmup-updates ${warmup}  --total-num-update ${total_updates} --max-update ${total_updates}"
elif [ ${scheduler} = "tristage" ]; then
scheduler_str="--lr ${lr} --lr-scheduler tri_stage --warmup-steps ${warmup} --hold-steps ${hs} --decay-steps ${ds} --init-lr-scale 1e-2 --final-lr-scale 1e-2  --max-update ${total_updates}"
fi

export expname=iwslt-${opt_str}-${scheduler}-hs${hs}-ds${ds}-lr${lr}-wd${wd}-warm${warmup}-eps${eps}${others_print}-seed${seed}-v2

export CUDA_VISIBLE_DEVICES=${gpu}

bleu_args="'{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}'"

cmd="
python train.py
    data-bin/iwslt14.tokenized.de-en
    --save-dir chks/${expname}
    --log-interval 10
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed
    --optimizer adam --adabound True --clip-norm 0.0 --use-old-adam --gamma ${gamma} --final-lr ${final_lr}
    ${scheduler_str}
    --dropout 0.3 --weight-decay ${wd}
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1
    --max-tokens ${tokens}
    --eval-bleu
    --keep-last-epochs 0 --no-save-optimizer-state
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
cmd="${cmd} --comet --api-key OVyeR5Xs0GGidqMc8g1s9K3Lx --comet-project 'iwslt-de-en-neurips' --comet-real-tag neurips --comet-tag ${expname}
> ${logpath} 2>&1
"
echo ${logpath}
fi

#echo ${cmd}
eval ${cmd}

}
wd=1e-2

#source ~/cmlscratch/anaconda3/etc/profile.d/conda.sh
#conda activate base
#
seed_list=(65443 9876 5454 3234 4543112)
flr_list=( 0.1  )
gamma_list=(1e-7 1e-8)
#
sidx=$(( (${SLURM_ARRAY_TASK_ID} - 1) % 5 ))
lidx=0   #$(( (${SLURM_ARRAY_TASK_ID} - 1) / 5 % 1  ))
gidx=$(( (${SLURM_ARRAY_TASK_ID} - 1) / 5 % 2 ))
#
## please refer to hyper parameter settings in the appendix
#
seed="${seed_list[sidx]}"
flr="${flr_list[lidx]}"
gamma="${gamma_list[gidx]}"
##
##
runexp    0   5e-4  ${gamma}     ${flr}        4000   1e-7   tristage       32000  24000  ${wd}  ${seed}

# runexp gpu   lr   gamma   final_lr    warmup   eps   scheduler       hs     ds    wd   seed  flags
#runexp    0   5e-4  1e-6     0.1        4000   1e-7   tristage       32000  24000  ${wd}  65443
#runexp    2   5e-4  1e-6     0.1        4000   1e-7   tristage       32000  24000  ${wd}  4543112

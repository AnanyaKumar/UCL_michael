sbatch -p jag-hi --job-name=cifar10_sup_r18_0 --output=sl_outs/cifar10_sup_r18_0 scripts/run_sbatch.sh "python main.py --data_dir=../Data/ --log_dir=../logs/ -c=configs/sup_finetune_c10.yaml --hide_progress=True --group_name=cifar10_sup_r18 --run_name=t0 --model.backbone=resnet18"

sbatch -p jag-hi --job-name=cifar10_sup_r152_0 --output=sl_outs/cifar10_sup_r152_0 scripts/run_sbatch.sh "python main.py --data_dir=../Data/ --log_dir=../logs/ -c=configs/sup_finetune_c10.yaml --hide_progress=True --group_name=cifar10_sup_r152 --run_name=t0 --model.backbone=resnet152"

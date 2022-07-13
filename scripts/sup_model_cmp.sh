# sbatch -p jag-hi --job-name=cifar10_sup_r18_0 --output=sl_outs/cifar10_sup_r18_0 scripts/run_sbatch.sh "python main.py --data_dir=../Data/ --log_dir=../logs/ -c=configs/sup_finetune_c10.yaml --hide_progress=True --group_name=cifar10_sup_r18 --run_name=t0 --model.backbone=resnet18"

# sbatch -p jag-hi --job-name=cifar10_sup_r152_0 --output=sl_outs/cifar10_sup_r152_0 scripts/run_sbatch.sh "python main.py --data_dir=../Data/ --log_dir=../logs/ -c=configs/sup_finetune_c10.yaml --hide_progress=True --group_name=cifar10_sup_r152 --run_name=t0 --model.backbone=resnet152"

# sbatch -p jag-hi --job-name=cifar10_sup_r152_lr0.1 --output=sl_outs/cifar10_sup_r152_lr0.1 scripts/run_sbatch.sh "python main.py --data_dir=../Data/ --log_dir=../logs/ -c=configs/sup_finetune_c10.yaml --hide_progress=True --group_name=cifar10_sup_r152 --run_name=lr0.1 --model.backbone=resnet152 --train.base_lr=0.1"

#sbatch -p jag-hi --job-name=cifar10_sup_r152_lr0.01 --output=sl_outs/cifar10_sup_r152_lr0.01 scripts/run_sbatch.sh "python main.py --data_dir=../Data/ --log_dir=../logs/ -c=configs/sup_finetune_c10.yaml --hide_progress=True --group_name=cifar10_sup_r152 --run_name=lr0.01 --model.backbone=resnet152 --train.base_lr=0.01"

# sbatch -p jag-hi --job-name=cifar10_sup_r152_lr0.3 --output=sl_outs/cifar10_sup_r152_lr0.3 scripts/run_sbatch.sh "python main.py --data_dir=../Data/ --log_dir=../logs/ -c=configs/sup_finetune_c10.yaml --hide_progress=True --group_name=cifar10_sup_r152 --run_name=lr0.3 --model.backbone=resnet152 --train.base_lr=0.3"

# sbatch -p jag-hi --job-name=cifar10_sup_r152_lr1.0 --output=sl_outs/cifar10_sup_r152_lr1.0 scripts/run_sbatch.sh "python main.py --data_dir=../Data/ --log_dir=../logs/ -c=configs/sup_finetune_c10.yaml --hide_progress=True --group_name=cifar10_sup_r152 --run_name=lr1.0 --model.backbone=resnet152 --train.base_lr=1.0"

sbatch -p jag-hi --job-name=cifar10_sup_r18_no_augs --output=sl_outs/cifar10_sup_r18_no_augs scripts/run_sbatch.sh "python main.py --data_dir=../Data/ --log_dir=../logs/ -c=configs/sup_finetune_c10.yaml --hide_progress=True --group_name=cifar10_sup_r18_no_augs --run_name=t0 --model.backbone=resnet18 --no_train_augs=True"

# sbatch -p jag-hi --job-name=cifar10_sup_r18_subsampled --output=sl_outs/cifar10_sup_r18_subsampled scripts/run_sbatch.sh "python main.py --data_dir=../Data/ --log_dir=../logs/ -c=configs/sup_finetune_c10.yaml --hide_progress=True --group_name=cifar10_sup_r18_subsampled --run_name=t0 --model.backbone=resnet18 --train_len=10000 --train.num_epochs=250 --train.stop_at_epoch=250"



# BD_Assignment

## 1. 安装依赖
pip install -r requirements.txt

## 2. 快速测试（验证代码可运行）
bash run_quick_test.sh

## 3. 运行完整实验
bash run_all_experiments.sh

## 4. 单独运行某个实验
python train_single_gpu.py --model resnet50 --batch_size 128 --epochs 50 --amp
python train_ddp.py --model resnet50 --batch_size 128 --epochs 50 --backend nccl --scale_lr

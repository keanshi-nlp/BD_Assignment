# BD_Assignment

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 快速测试（验证代码可运行）

```bash
bash ./scripts/run_quick_test.sh
```

## 3. 运行实验

```bash
bash ./scripts/run_all_experiments.sh 
bash ./scripts/run_reduce.sh
```

## 4. 单独运行某个实验

```bash
python ./src/train_single_gpu.py --model resnet50 --batch_size 128 --epochs 50 --amp
python ./src/train_ddp.py --model resnet50 --batch_size 128 --epochs 50 --backend nccl --scale_lr
```

## 5. 使用 nsight 分析结果
```bash
bash ./scripts/run_ntvx.sh 
```
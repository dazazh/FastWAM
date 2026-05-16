# Conda Guide for FastWAM

This project uses a manually initialized Conda setup on this server.

## 1) Initialize Conda in each new shell

Run these commands every time you open a new terminal:

```bash
source /workspace/zhengxj2@xiaopeng.com/workspace/tools/my_env.sh
source /dataset_rc_mm/zhengxj2@xiaopeng.com/softwares/miniconda3/etc/profile.d/conda.sh
```

Do **not** run `conda init` on this cluster.

## 2) Activate the project environment

```bash
conda activate fastwam
```

To leave the environment:

```bash
conda deactivate
```

## 3) First-time setup (if `fastwam` does not exist yet)

From the project root (`FastWAM/`):

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --no-deps \
  -i http://mirrors.cloud.aliyuncs.com/pypi/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --trusted-host mirrors.cloud.aliyuncs.com \
  --trusted-host download.pytorch.org
pip install -e . \
  -i http://mirrors.cloud.aliyuncs.com/pypi/simple/ \
  --trusted-host mirrors.cloud.aliyuncs.com \
  --trusted-host nexus-wl.xiaopeng.link
```

## 4) Quick verification

```bash
python - <<'PY'
import torch, torchvision, fastwam
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("fastwam import: ok")
print("cuda available:", torch.cuda.is_available())
PY
```

## 5) Common issues

- **`conda: command not found`**  
  You forgot to source Conda:
  `source /dataset_rc_mm/zhengxj2@xiaopeng.com/softwares/miniconda3/etc/profile.d/conda.sh`

- **Downloads are very slow or appear stuck**  
  Use the Aliyun mirror as shown above (`-i http://mirrors.cloud.aliyuncs.com/pypi/simple/`).

- **`cuda available: False` with driver warning**  
  Your GPU driver may be older than required by CUDA 12.8 wheels. In that case either upgrade the driver or install a PyTorch build compatible with your current driver.


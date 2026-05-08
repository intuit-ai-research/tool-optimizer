## Environment Setup

### 1. Install the CUDA
Check this [link](https://verl.readthedocs.io/en/latest/start/install.html) for more details.

Quick setup if you restart the SageMaker instance:
```bash
# At where the file is downloaded
sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
sudo update-alternatives --set cuda /usr/local/cuda-12.8
```

```bash
# At where the file is downloaded
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12
```

Check if CUDA is 12.8:
```bash
nvcc --version
```

If not, run the following command:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && export PATH=/usr/local/cuda/bin:$PATH && nvcc --version
```


### 2. Install the python dependencies

### 2.1 Use [environment.yml](environment.yml)
```bash
conda env create -f environment.yml
conda activate agent_train
```

### 2.2 Install based on the docs
```bash
conda create -n verl python==3.12
conda activate verl
```

Then, execute the install.sh script that we provided in verl:
```bash
# Make sure you have activated verl conda env
# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

### 3. Install the verl
```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install --no-deps -e .
```

Currently, Verl is at commit b9bd00efba253ea90072555c45692054cf703de2.


### 4. Install the smolagent
```bash
pip install "smolagents[toolkit,openai,telemetry,vllm]>=1.23.0" "tqdm>=4.67.1" "termcolor>=3.2.0" "rank-bm25>=0.2.2" "tenacity>=9.1.2" "joblib>=1.5.2"
```

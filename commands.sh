# Step 1: Request interactive GPU session (run this on the login node)
sinteractive -A gpu --nodes=1 --gres=gpu:1 -t 04:00:00

# Step 2: Load CUDA module (run once inside the GPU session)
module load cuda

# Step 3: Pull latest code
cd ~/ece695_hml-p1
git pull

# Step 4: Compile
make

# Step 5: Run naive GEMM sanity check
./gemm_naive

# Step 6: Run profiler
python3 profiler.py

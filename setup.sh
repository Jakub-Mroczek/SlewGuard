# Step 1: Request interactive GPU session (run this on the login node)
sinteractive -A gpu --nodes=1 --gres=gpu:1 -t 04:00:00

# Step 2: Load CUDA module (run once inside the GPU session)
module load cuda

# Step 3: Pull latest code
cd ~/ece695_hml-p1
git pull

# Step 4: Install Python dependencies
pip install -r requirements.txt

# Step 5: Compile
make

# Step 6: Run profiler
python3 profiler.py

# Example output
# gemm_lib.cu
#   batch_size  samples  avg_power_W  max_power_W  avg_util%  avg_clock_MHz
# ----------------------------------------------------------------------
#           64       18         36.9         45.6        2.0          930.4
#          128        9         47.0         47.7       18.0         1245.0
#          256       10         50.2         54.0       31.0         1245.0
#          512       16         55.9         59.5       33.1         1245.0
#         1024       23         60.9         68.9       39.6         1264.3

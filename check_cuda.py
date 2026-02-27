import torch
import sys

print("--- System Info ---")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

print("\n--- CUDA Info ---")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device Index: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # Simple tensor test
    print("\n--- Running Tensor Test ---")
    try:
        x = torch.rand(5, 3).cuda()
        print("Tensor successfully moved to GPU!")
        print(f"Tensor Device: {x.device}")
        
        # Matrix multiplication test
        y = torch.matmul(x, x.transpose(0, 1))
        print("Matrix multiplication on A100: Success")
    except Exception as e:
        print(f"Tensor test failed: {e}")
else:
    print("\n[!] WARNING: CUDA is NOT available to PyTorch.")
    print("Check if you started the container with --gpus all")

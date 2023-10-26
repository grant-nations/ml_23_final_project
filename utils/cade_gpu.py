import os

## This function locates an available gpu for usage. In addition, this function reserves a specificed
## memory space exclusively for your account. The memory reservation prevents the decrement in computational
## speed when other users try to allocate memory on the same gpu in the shared systems, i.e., CADE machines. 
## Note: If you use your own system which has a GPU with less than 4GB of memory, remember to change the 
## specified mimimum memory.
import subprocess

def define_gpu_to_use(minimum_memory_mb = 3500):
    thres_memory = 600
    gpu_to_use = None
    
    try: 
        os.environ['CUDA_VISIBLE_DEVICES']
        print('GPU already assigned before: ' + str(os.environ['CUDA_VISIBLE_DEVICES']))
        return
    except:
        pass
    
    for i in range(16):
        cmd = ["nvidia-smi", "--query-gpu=memory.free", f"-i {i}", "--format=csv,nounits,noheader"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, shell=True)
        free_memory = int(result.stdout.strip())
        
        if free_memory > minimum_memory_mb - thres_memory:
            gpu_to_use = i
            break
            
    if gpu_to_use is None:
        print(f'Could not find any GPU available with the required free memory of {minimum_memory_mb}MB. Please use a different system for this assignment.')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_to_use)
        print(f'Chosen GPU: {gpu_to_use}')

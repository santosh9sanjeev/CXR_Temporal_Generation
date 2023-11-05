import subprocess

def get_gpu_pids(gpu_id):
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', f'--id={gpu_id}', '--format=csv,noheader'], capture_output=True, text=True)
        pids = [int(pid) for pid in result.stdout.strip().split('\n')]
        return pids
    except Exception as e:
        print(f"Error: {e}")
        return []

def kill_processes(pids):
    try:
        for pid in pids:
            subprocess.run(['kill', str(pid)])
            print(f"Process with PID {pid} killed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    gpu_id = 6  # Change this to the GPU ID you're interested in

    # Get PIDs associated with GPU
    pids = get_gpu_pids(gpu_id)

    if pids:
        # Kill the processes
        kill_processes(pids)
    else:
        print(f"No processes found on GPU {gpu_id}.")

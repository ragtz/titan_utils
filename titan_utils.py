#####################
# NVIDIA GPU STUFF
#####################

import subprocess, re
import numpy as np
import argparse

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

def nvidia_smi(idx=None, args=None):
    if idx is None and args is None:
        return "nvidia-smi"
    elif idx is None and not args is None:
        return "nvidia-smi " + args
    elif not idx is None and args is None:
        return "titan-" + str(idx) + " 'nvidia-smi'"
    else:
        return "titan-" + str(idx) + " 'nvidia-smi" + args + "'"

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus(idx=None):
    """Returns list of available GPU ids."""
    output = run_command(nvidia_smi(idx, " -L"))
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map(idx=None):
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command(nvidia_smi(idx))
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |     0        8734   C   python                                                                        11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus(idx)}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory(idx):
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map(idx).items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu

def pick_titan_lowest_mean_memory(s_idx=1, e_idx=15):
    mean_memory_map = [(np.mean(gpu_memory_map(idx).values()), idx) for idx in range(s_idx, e_idx+1)]
    best_memory, best_titan = sorted(mean_memory_map)[0]
    return best_titan

if __name__=="__main__":
    parser = argparse.ArgumentParser("Utility to select most available machine and gpu in Titan cluster")
    parser.add_argument("-s", metavar="INT", type=int, default=1, required=False, help="start titan index")
    parser.add_argument("-e", metavar="INT", type=int, default=15, required=False, help="end titan index")

    args = parser.parse_args()
    s = args.s
    e = args.e

    best_titan = pick_titan_lowest_mean_memory(s,e)
    best_gpu = pick_gpu_lowest_memory(best_titan)

    print("titan-"+str(best_titan))
    print("gpu-"+str(best_gpu))


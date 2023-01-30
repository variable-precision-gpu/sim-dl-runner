from collections import OrderedDict
import os

BASELINE_RESULTS = "./programs/MLP/cuda_training_gpgpu_vf32_full_inference.txt"
REDUCED_PREC_RESULTS = "./programs/MLP/cuda_training_gpgpu_vf32_8_inference.txt"


deterioration = []
improvement = []

with open(BASELINE_RESULTS, "r") as baseline, open(REDUCED_PREC_RESULTS, "r") as reduced:
    for i, (line1, line2) in enumerate(zip(baseline, reduced), start=1):
        if line1 != line2:
            # deterioration
            if line1.rstrip() == "1":
              deterioration.append(i)
            # improvement
            else:
              improvement.append(i)

print("{} detoriorations: {}".format(len(deterioration), deterioration))
print("{} improvements: {}".format(len(improvement), improvement))
# print(same)
# print(different)

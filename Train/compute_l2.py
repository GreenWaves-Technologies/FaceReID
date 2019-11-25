import numpy as np
import torch

file1 = '/home/maxim/work/face_re_id/office_cropped/maxim/IMG_20190412_134107.json'
file2 = '/home/maxim/work/face_re_id/office_cropped/maxim/IMG_20190412_143619.json'
res_file = '/home/maxim/work/face_re_id/office_cropped/maxim/result.json'

with open(file1) as f:
    vec1 = np.array(eval(f.read()))
print(vec1)
with open(file2) as f:
    vec2 = np.array(eval(f.read()))

print(vec2)
pow2_1 = np.power(vec1, 2).sum()
pow2_2 = np.power(vec2, 2).sum()

mult = np.multiply(vec1, vec2).sum()
print(pow2_1, pow2_2, mult)
res = pow2_2 + pow2_1 - 2 * mult
with open(res_file, 'w') as f:
    print(res, file=f)
# distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#         distmat.addmm_(1, -2, qf, gf.t())

qf = torch.from_numpy(vec1)
gf = torch.from_numpy(vec2)
m, n = qf.size(0), gf.size(0)

distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
          torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
distmat.addmm_(1, -2, qf, gf.t())

print(distmat.data.numpy()[0, 0])
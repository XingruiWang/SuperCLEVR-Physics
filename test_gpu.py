import torch
import bpy

a = torch.tensor([0])
print(a.device)
device = torch.device("cuda:0")
a.to('cuda', non_blocking=True)
a= a.cuda()
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
print(a.device)

import pdb; pdb.set_trace()

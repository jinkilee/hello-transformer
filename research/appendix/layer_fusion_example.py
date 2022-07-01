# Reference: https://nenadmarkus.com/p/fusing-batchnorm-and-conv/

import torch
import torchvision
import time

torch.set_grad_enabled(False)
x = torch.randn(16, 3, 256, 256)
rn18 = torchvision.models.resnet18(pretrained=True)
rn18.eval()
net = torch.nn.Sequential(
    rn18.conv1,
    rn18.bn1
)


def fuse_conv_and_bn(conv, bn):
    #
    # init
    fusedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    print(f'in_channels={conv.in_channels} out_channels={conv.out_channels} kernel_size={conv.kernel_size}')
    #
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
    fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
    print(f'w_conv: {conv.weight.clone().shape} -> {w_conv.shape}')
    print(f'w_bn: {bn.weight.shape} -> {w_bn.shape}')
    #
    # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros( conv.weight.size(0) )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_( torch.matmul(w_bn, b_conv) + b_bn )

    print(fusedconv.weight.shape, fusedconv.bias.shape)

    #
    # we're done
    return fusedconv



y1 = net.forward(x)
fusedconv = fuse_conv_and_bn(net[0], net[1])
y2 = fusedconv.forward(x)
d = (y1 - y2).norm().div(y1.norm()).item()
print("error: %.8f" % d)

t1_start = time.time()
for _ in range(100):
    net.forward(x)
t1_end = time.time()
print(f't1={(t1_end - t1_start)/100}ms')

t2_start = time.time()
for _ in range(100):
    fusedconv.forward(x)
t2_end = time.time()
print(f't2={(t2_end - t2_start)/100}ms')





import torch

import utils_cuda


def main():
    input = torch.arange(120).type(torch.float).to(device='cuda').contiguous()
    output = torch.zeros(120).type(torch.int32).to(device='cuda').contiguous()
    print(input)
    utils_cuda.float2bin(input, output)
    print(output)
    compressor = torch.as_tensor(0xff800000).type(torch.int32).to(device='cuda')
    output &= compressor
    print(output)
    output2 = torch.zeros(120).type(torch.float).to(device='cuda').contiguous()
    utils_cuda.bin2float(output, output2)
    print(output2)


if __name__ == '__main__':
    main()

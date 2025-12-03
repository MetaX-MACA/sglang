import json
import numpy as np
import sys
import os

def parser_mcTracer(input_json, output_txt):    
    deepspeed = True
    with open(input_json, 'r') as f:
        data = json.load(f)
    f.close()

    info_list = data['traceEvents'][1:]

    kernel_list = []
    ts_list = []
    for info in info_list:
        # pi-> device  1: cpu 2: gpu 3: DMA
        # cat or queue_id --> stream
        # if 'cat' in info:
            # if info['pid'] == 2 or info['cat'] == 'kernel':
        try:
            kernel_list.append([info['name'], info['args']['grid'], info['args']['block'], info['args']['device'], info['dur']])
        except:
                # print('kernel {} has no duration, skiped~'.format(info['name']))
            continue

        ts_list.append(info['ts'])
    # sort kernel by timestamp
    ts = np.array(ts_list)
    ts_sorted_idx = np.argsort(ts).tolist()

    print(len(ts_list))

    time_str = 'us'

    f = open(output_txt, 'w')

    # for kernel, duration in kernel_list:
    for idx in ts_sorted_idx:
        kernel, grid, block, device_id, duration = kernel_list[idx]
        end_idx = kernel.find('<')
        if end_idx < 0:
            end_idx = kernel.find('(')
        ArgMaxOps = "ArgMaxOps" in kernel
        flashinfer = "flashinfer" in kernel
        triton = "triton" in kernel
        kernel = kernel[:end_idx]
        if kernel[:4] == 'void':
            kernel = kernel[5:]

        if ArgMaxOps and "ArgMaxOps" not in kernel:
            kernel ="{}.ArgMaxOps".format(kernel)
        if flashinfer and "flashinfer" not in kernel:
            kernel ="flashinfer_{}".format(kernel)
        if triton and "triton" not in kernel:
            kernel ="triton_{}".format(kernel)
        if deepspeed:
            grid = '{},{},{}'.format(grid[0], grid[1], grid[2])
            block = '{},{},{}'.format(block[0], block[1], block[2])
            duration = float(duration)
        else:
            grid = '{},{},{}'.format(grid['x'], grid['y'], grid['z'])
            block = '{},{},{}'.format(block['x'], block['y'], block['z'])    
            duration = float(duration) / 1000
        f.write('{},{:.3f} {},{}\n'.format(kernel, duration, time_str, device_id))
        
    f.close()


if __name__ == '__main__':
    file_in=sys.argv[1]
    file_list = []
    if os.path.isdir(file_in):
        for json_file in os.listdir(file_in):
            if json_file[-5:] == '.json':
                file_list.append(os.path.join(file_in, json_file))
    elif file_in[-5:] == '.json':
        file_list.append(file_in)
    else:
        print('input file is not a mctracer result json. Or input file path not contains mctracer json')
        sys.exit()
    for json_file in file_list:
        output_txt=json_file[:-5] + '.txt'
        print(f'Start analysis {json_file} ...')
        parser_mcTracer(json_file, output_txt)
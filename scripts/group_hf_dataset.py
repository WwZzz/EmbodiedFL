import h5py
import numpy as np
import argparse

# file_path = "/home/dataset/wz/robomimic_data/tool_hang/ph/low_dim_v15.hdf5"
source_group = "/data"
mask_group = "/mask"
# group_num = 100

parser = argparse.ArgumentParser()
parser.add_argument('--file', help='save_dir', type=str, default='data/robomimic/tool_hang/ph/low_dim_v15.hdf5')
parser.add_argument('--val', help='ratio of train-val', type=float, default=0.2)
parser.add_argument('--num_groups', help='num_workers', type=int, default=20)
args = parser.parse_args()

with h5py.File(args.file, "a") as f:
    if mask_group in f: del f[mask_group]
    mask = f.create_group(mask_group)
    members = sorted(f[source_group].keys(), key=lambda x: int(x.split('_')[1]))
    data_size = len(members)
    if args.num_groups<=0:
        group_size = data_size
        if args.val<0 or group_size<=1:
            exit()
        val_size = max(int(group_size*args.val), 1)
        train_size = group_size - val_size
        grouped = [members[:train_size], members[train_size:]]
        group_names = ['train', 'valid']
    else:
        group_size = data_size // args.num_groups
        # if data_size % args.num_groups != 0: group_size += 1
        tmp_grouped = [members[i * group_size:(i + 1) * group_size] for i in range(args.num_groups)]
        rest_num = data_size % args.num_groups
        if rest_num != 0:
            rest = members[-rest_num:]
            for mi, tgi in zip(rest, tmp_grouped):
                tgi.append(mi)
        if args.val<0:
            grouped = tmp_grouped
            group_names = [f'{i}' for i in range(len(tmp_grouped))]
        else:
            grouped = []
            group_names = []
            for i, gim in enumerate(tmp_grouped):
                assert len(gim)>1
                val_size = max(int(len(gim)*args.val), 1)
                train_size = group_size - val_size
                grouped.extend([gim[:train_size], gim[train_size:]])
                group_names.extend([f'train_{i}', f'valid_{i}'])
    str_dtype = h5py.string_dtype(encoding='ascii')  # 强制ASCII编码
    for group_idx, (sub_members, gname) in enumerate(zip(grouped, group_names)):
        validated = []
        for name in sub_members:
            try:
                # 严格ASCII验证
                name.encode('ascii')
                validated.append(name)
            except UnicodeEncodeError:
                # 替换非法字符为下划线
                safe_name = ''.join(c if ord(c) < 128 else '_' for c in name)
                validated.append(safe_name)
            # 转换为numpy字节数组
            byte_array = np.array([np.bytes_(s) for s in validated], dtype=object)
            # 创建数据集
        dset = mask.create_dataset(
            name=gname,
            data=byte_array,
            dtype=str_dtype,
            track_times=False  # 禁用时间追踪提升性能
        )
        # 添加编码元数据
        dset.attrs.create('ENCODING', 'ascii')


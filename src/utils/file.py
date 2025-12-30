import os.path as osp
import pandas as pd
import numpy as np
import os
from utils import Center

# def load_csv_tennis(csv_path, visible_flags, frame_dir=None):
#     df = pd.read_csv(csv_path)
#     fnames, visis, xs, ys = df['file name'].tolist(), df['visibility'].tolist(), df['x-coordinate'].tolist(), df['y-coordinate'].tolist()
#     xyvs = {}
#     for fname, visi, x, y in zip(fnames, visis, xs, ys):
#         fid = int(osp.splitext(fname)[0])
#         if frame_dir is not None:
#             frame_path = osp.join(frame_dir, fname)
#         else:
#             frame_path = None

#         if fid in xyvs.keys():
#             raise KeyError('fid {} already exists'.format(fid ))

#     xyvs = {}
#     for fname, visi, x, y in zip(fnames, visis, xs, ys):
#         fid = int(osp.splitext(fname)[0])
#         if frame_dir is not None:
#             frame_path = osp.join(frame_dir, fname)
#         else:
#             frame_path = None

#         if fid in xyvs.keys():
#             raise KeyError('fid {} already exists'.format(fid ))
        
#         if np.isnan(x) or np.isnan(y):
#             if (int(visi) in visible_flags):
#                 print(visible_flags)
#                 print(fname)
#                 print(visi, x, y)
#                 print(int(visi))
#                 quit()

#         xyvs[fid] = {'center': Center(x=float(x),
#                                       y=float(y),
#                                       is_visible=True if int(visi) in visible_flags else False,
#                                ),
#                      'file_name': fname,
#                      'frame_path': frame_path
#                      }

#     return xyvs

# 能够处理不存在或为空的CSV文件
# def load_csv_tennis(csv_path, visible_flags, frame_dir=None):
#     # 检查CSV文件是否存在且不为空
#     if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
#         # 如果CSV文件不存在或为空，返回一个空字典
#         if frame_dir:
#             # 创建一个基于frame_dir中所有图像的空字典
#             xyvs = {}
#             for fname in os.listdir(frame_dir):
#                 if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
#                     fid = int(os.path.splitext(fname)[0])
#                     xyvs[fid] = {'center': Center(x=0.0,
#                                                   y=0.0,
#                                                   is_visible=False,
#                                            ),
#                                  'file_name': fname,
#                                  'frame_path': os.path.join(frame_dir, fname)
#                                  }
#             return xyvs
#         else:
#             return {}

#     df = pd.read_csv(csv_path)
#     fnames, visis, xs, ys = df['file name'].tolist(), df['visibility'].tolist(), df['x-coordinate'].tolist(), df['y-coordinate'].tolist()
#     xyvs = {}
#     for fname, visi, x, y in zip(fnames, visis, xs, ys):
#         fid = int(os.path.splitext(fname)[0])
#         if frame_dir is not None:
#             frame_path = os.path.join(frame_dir, fname)
#         else:
#             frame_path = None

#         if fid in xyvs.keys():
#             raise KeyError('fid {} already exists'.format(fid ))
        
#         if np.isnan(x) or np.isnan(y):
#             if (int(visi) in visible_flags):
#                 print(visible_flags)
#                 print(fname)
#                 print(visi, x, y)
#                 print(int(visi))
#                 quit()

#         xyvs[fid] = {'center': Center(x=float(x),
#                                       y=float(y),
#                                       is_visible=True if int(visi) in visible_flags else False,
#                                ),
#                      'file_name': fname,
#                      'frame_path': frame_path
#                      }

#     return xyvs


def load_csv_tennis(csv_path, visible_flags, frame_dir=None):
    # 检查CSV文件是否存在且不为空
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        # 如果CSV文件不存在或为空，返回一个空字典
        if frame_dir:
            # 创建一个基于frame_dir中所有图像的空字典
            xyvs = {}
            for fname in os.listdir(frame_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # 提取文件名中的数字部分
                    name_without_ext = os.path.splitext(fname)[0]
                    # 使用正则表达式提取数字
                    import re
                    numbers = re.findall(r'\d+', name_without_ext)
                    if numbers:
                        fid = int(numbers[-1])  # 使用最后一个数字组
                    else:
                        raise ValueError(f"无法从文件名 {name_without_ext} 中提取帧ID")
                    xyvs[fid] = {'center': Center(x=0.0,
                                                  y=0.0,
                                                  is_visible=False,
                                           ),
                                 'file_name': fname,
                                 'frame_path': os.path.join(frame_dir, fname)
                                 }
            return xyvs
        else:
            return {}

    df = pd.read_csv(csv_path)
    fnames, visis, xs, ys = df['file name'].tolist(), df['visibility'].tolist(), df['x-coordinate'].tolist(), df['y-coordinate'].tolist()
    xyvs = {}
    for fname, visi, x, y in zip(fnames, visis, xs, ys):
        # 提取文件名中的数字部分
        name_without_ext = os.path.splitext(fname)[0]
        # 使用正则表达式提取数字
        import re
        numbers = re.findall(r'\d+', name_without_ext)
        if numbers:
            fid = int(numbers[-1])  # 使用最后一个数字组
        else:
            raise ValueError(f"无法从文件名 {name_without_ext} 中提取帧ID")
        
        if frame_dir is not None:
            frame_path = os.path.join(frame_dir, fname)
        else:
            frame_path = None

        if fid in xyvs.keys():
            raise KeyError('fid {} already exists'.format(fid ))
        
        if np.isnan(x) or np.isnan(y):
            if (int(visi) in visible_flags):
                print(visible_flags)
                print(fname)
                print(visi, x, y)
                print(int(visi))
                quit()

        xyvs[fid] = {'center': Center(x=float(x),
                                      y=float(y),
                                      is_visible=True if int(visi) in visible_flags else False,
                               ),
                     'file_name': fname,
                     'frame_path': frame_path
                     }

    return xyvs
import os
import os.path as osp
import logging
from omegaconf import DictConfig
from tqdm import tqdm
import pandas as pd
import numpy as np

from utils import Center
from utils import load_csv_tennis as load_csv
from utils import refine_gt_clip_tennis as refine_gt_clip

log = logging.getLogger(__name__)

def get_clips(cfg, train_or_test='test', gt=True):
    root_dir      = cfg['dataset']['root_dir']
    matches       = cfg['dataset'][train_or_test]['matches']
    csv_filename  = cfg['dataset']['csv_filename']
    ext           = cfg['dataset']['ext']
    visible_flags = cfg['dataset']['visible_flags']

    clip_dict = {}
    for match in matches:
        match_video_dir = osp.join(root_dir, match)
        clip_names     = os.listdir(match_video_dir)
        clip_names.sort()
        for clip_name in clip_names:
            clip_dir      = osp.join(root_dir, match, clip_name)
            clip_csv_path = osp.join(root_dir, match, clip_name, csv_filename)
            frame_names = []
            for frame_name in os.listdir(clip_dir):
                if frame_name.endswith(ext):
                    frame_names.append(frame_name)
            frame_names.sort()
            ball_xyvs = load_csv(clip_csv_path, visible_flags) if gt else None
            clip_dict[(match, clip_name)] = {'clip_dir_or_path': clip_dir, 'clip_gt_dict': ball_xyvs, 'frame_names': frame_names}

    return clip_dict

class Tennis(object):
    def __init__(self, 
                 cfg: DictConfig,
    ):
        self._root_dir             = cfg['dataset']['root_dir']
        self._ext                  = cfg['dataset']['ext']
        self._csv_filename         = cfg['dataset']['csv_filename']
        self._visible_flags        = cfg['dataset']['visible_flags']
        self._train_matches        = cfg['dataset']['train']['matches']
        self._test_matches         = cfg['dataset']['test']['matches']
        self._train_num_clip_ratio = cfg['dataset']['train']['num_clip_ratio']
        self._test_num_clip_ratio  = cfg['dataset']['test']['num_clip_ratio']

        self._train_refine_npz_path = cfg['dataset']['train']['refine_npz_path']
        self._test_refine_npz_path  = cfg['dataset']['test']['refine_npz_path']

        self._frames_in  = cfg['model']['frames_in']
        self._frames_out = cfg['model']['frames_out']
        self._step       = cfg['detector']['step']

        self._load_train      = cfg['dataloader']['train']
        self._load_test       = cfg['dataloader']['test']
        self._load_train_clip = cfg['dataloader']['train_clip']
        self._load_test_clip  = cfg['dataloader']['test_clip']

        self._train_all        = []
        self._train_clips      = {}
        self._train_clip_gts   = {}
        self._train_clip_disps = {}
        if self._load_train or self._load_train_clip:
            train_outputs = self._gen_seq_list(self._train_matches, self._train_num_clip_ratio, self._train_refine_npz_path)
            self._train_all                = train_outputs['seq_list'] 
            self._train_num_frames         = train_outputs['num_frames']
            self._train_num_frames_with_gt = train_outputs['num_frames_with_gt']
            self._train_num_matches        = train_outputs['num_matches']
            self._train_num_rallies        = train_outputs['num_rallies']
            self._train_disp_mean          = train_outputs['disp_mean']
            self._train_disp_std           = train_outputs['disp_std']
            if self._load_train_clip:
                self._train_clips      = train_outputs['clip_seq_list_dict']
                self._train_clip_gts   = train_outputs['clip_seq_gt_dict_dict']
                self._train_clip_disps = train_outputs['clip_seq_disps']

        self._test_all        = []
        self._test_clips      = {}
        self._test_clip_gts   = {}
        self._test_clip_disps = {}
        if self._load_test or self._load_test_clip:
            test_outputs  = self._gen_seq_list(self._test_matches, self._test_num_clip_ratio, self._test_refine_npz_path)
            self._test_all                 = test_outputs['seq_list']
            self._test_num_frames          = test_outputs['num_frames']
            self._test_num_frames_with_gt  = test_outputs['num_frames_with_gt']
            self._test_num_matches         = test_outputs['num_matches']
            self._test_num_rallies         = test_outputs['num_rallies']
            self._test_disp_mean           = test_outputs['disp_mean']
            self._test_disp_std            = test_outputs['disp_std']
            if self._load_test_clip:
                self._test_clips               = test_outputs['clip_seq_list_dict']
                self._test_clip_gts            = test_outputs['clip_seq_gt_dict_dict']
                self._test_clip_disps          = test_outputs['clip_seq_disps']

        # show stats
        log.info('=> Tennis loaded' )
        log.info("Dataset statistics:")
        log.info("-------------------------------------------------------------------------------------")
        log.info("subset          | # batch | # frame | # frame w/ gt | # rally | # match | disp[pixel]")
        log.info("-------------------------------------------------------------------------------------")
        if self._load_train:
            log.info("train           | {:7d} | {:7d} | {:13d} | {:7d} | {:7d} | {:2.1f}+/-{:2.1f}".format(len(self._train_all), self._train_num_frames, self._train_num_frames_with_gt, self._train_num_rallies, self._train_num_matches, self._train_disp_mean, self._train_disp_std ) )
        if self._load_train_clip:
            num_items_all          = 0
            num_frames_all         = 0
            num_frames_with_gt_all = 0
            num_clips_all          = 0
            disps_all              = []
            for key, clip in self._train_clips.items():
                num_items  = len(clip)
                num_frames = 0
                for tmp in clip:
                    num_frames += len( tmp['frames'] )
                num_frames_with_gt = num_frames
                clip_name = '{}_{}'.format(key[0], key[1])
                disps     = np.array( self._train_clip_disps[key] )
                log.info("{} | {:7d} | {:7d} | {:13d} |         |         | {:2.1f}+/-{:2.1f}".format(clip_name, num_items, num_frames, num_frames_with_gt, np.mean(disps), np.std(disps) ))
            
                num_items_all          += num_items
                num_frames_all         += num_frames
                num_frames_with_gt_all += num_frames_with_gt
                disps_all.extend(disps)
                num_clips_all += 1
            log.info("all         | {:7d} | {:7d} | {:13d} | {:7d} |         | {:2.1f}+/-{:2.1f}".format(num_items_all, num_frames_all, num_frames_with_gt_all, num_clips_all, np.mean(disps_all), np.std(disps_all) ))
        if self._load_test:
            log.info("test            | {:7d} | {:7d} | {:13d} | {:7d} | {:7d} | {:2.1f}+/-{:2.1f}".format(len(self._test_all), self._test_num_frames, self._test_num_frames_with_gt, self._test_num_rallies, self._test_num_matches, self._test_disp_mean, self._test_disp_std) )
        if self._load_test_clip:
            num_items_all          = 0
            num_frames_all         = 0
            num_frames_with_gt_all = 0
            num_clips_all          = 0
            disps_all              = []
            for key, test_clip in self._test_clips.items():
                num_items  = len(test_clip)
                num_frames = 0
                for tmp in test_clip:
                    num_frames += len( tmp['frames'] )
                num_frames_with_gt = num_frames
                clip_name = '{}_{}'.format(key[0], key[1])
                disps     = np.array( self._test_clip_disps[key] )
                log.info("{} | {:7d} | {:7d} | {:13d} |         |         | {:2.1f}+/-{:2.1f}".format(clip_name, num_items, num_frames, num_frames_with_gt, np.mean(disps), np.std(disps) ))
            
                num_items_all          += num_items
                num_frames_all         += num_frames
                num_frames_with_gt_all += num_frames_with_gt
                disps_all.extend(disps)
                num_clips_all += 1
            log.info("all         | {:7d} | {:7d} | {:13d} | {:7d} |         | {:2.1f}+/-{:2.1f}".format(num_items_all, num_frames_all, num_frames_with_gt_all, num_clips_all, np.mean(disps_all), np.std(disps_all) ))
        log.info("-------------------------------------------------------------------------------------")

    def _gen_seq_list(self, 
                      matches, 
                      num_clip_ratio, 
                      refine_npz_path=None,
    ):
        # 如果 matches 是 'all'，则自动获取根目录下的所有子目录
        if matches == 'all' or matches == ['all']:
            matches = [d for d in os.listdir(self._root_dir) 
                      if os.path.isdir(os.path.join(self._root_dir, d))]
        
        if refine_npz_path is not None:
            log.info('refine gt ball positions with {}'.format(refine_npz_path))

        seq_list              = []
        clip_seq_list_dict    = {}
        clip_seq_gt_dict_dict = {}
        clip_seq_disps        = {}
        num_frames         = 0
        num_matches        = len(matches)
        num_rallies        = 0
        num_frames_with_gt = 0
        disps              = []
        for match in matches:
            match_clip_dir = osp.join(self._root_dir, match)
            clip_names     = os.listdir(match_clip_dir)
            clip_names.sort()
            clip_names = clip_names[:int(len(clip_names)*num_clip_ratio)]
            num_rallies += len(clip_names)
            for clip_name in clip_names:
                clip_seq_list    = []
                clip_seq_gt_dict = {}
                clip_frame_dir   = osp.join(self._root_dir, match, clip_name)
                
                # 检查目录是否存在
                if not osp.exists(clip_frame_dir):
                    log.warning(f"Clip directory does not exist: {clip_frame_dir}, skipping...")
                    continue
                
                clip_csv_path    = osp.join(self._root_dir, match, clip_name, self._csv_filename )
                try:
                    ball_xyvs = load_csv(clip_csv_path, self._visible_flags, frame_dir=clip_frame_dir)
                except Exception as e:
                    log.warning(f"Error loading CSV for {match}/{clip_name}: {e}, creating empty annotations...")
                    ball_xyvs = {}
                
                frame_names = []
                try:
                    for frame_name in os.listdir(clip_frame_dir):
                        if frame_name.endswith(self._ext):
                            frame_names.append(frame_name)
                    frame_names.sort()
                except FileNotFoundError:
                    log.warning(f"Frame directory does not exist: {clip_frame_dir}, skipping clip {match}/{clip_name}")
                    continue
                
                # 检查是否有足够的帧来构建序列，如果没有则跳过此片段
                if len(frame_names) < self._frames_in or len(ball_xyvs) < self._frames_in:
                    log.warning(f"Skipping clip {match}/{clip_name} due to insufficient frames: {len(frame_names)} frame files, {len(ball_xyvs)} annotations, required {self._frames_in}")
                    # 对于预测任务，即使帧数不足，我们也尝试处理尽可能多的帧
                    if len(frame_names) > 0 and len(ball_xyvs) == 0:
                        # 如果没有标注但是有图片，为预测创建基础字典
                        ball_xyvs = {}
                        for fname in frame_names:
                            name_without_ext = os.path.splitext(fname)[0]
                            try:
                                fid = int(name_without_ext)
                            except ValueError:
                                import re
                                numbers = re.findall(r'\d+', name_without_ext)
                                if numbers:
                                    fid = int(numbers[-1])
                                else:
                                    continue  # 跳过无法解析的文件名
                            
                            ball_xyvs[fid] = {'center': Center(x=0.0,
                                                                y=0.0,
                                                                is_visible=False,
                                                        ),
                                            'file_name': fname,
                                            'frame_path': osp.join(clip_frame_dir, fname)
                                            }
                    
                    # 再次检查是否仍不足以构建序列
                    if len(frame_names) < self._frames_in or len(ball_xyvs) < self._frames_in:
                        log.warning(f"Still insufficient frames for clip {match}/{clip_name}, skipping...")
                        continue
            
                num_frames         += len(frame_names)
                num_frames_with_gt += len(ball_xyvs)
                
                if refine_npz_path is not None:
                    ball_xyvs = refine_gt_clip(ball_xyvs, clip_frame_dir, frame_names, refine_npz_path)


                for i in range(len(frame_names) - self._frames_in + 1):
                    # 检查是否有对应的标注
                    names = frame_names[i:i+self._frames_in]
                    paths = [osp.join(clip_frame_dir, name) for name in names]
                    
                    # 为这一帧序列创建annos，如果缺少标注则使用默认值
                    annos = []
                    all_annos_available = True
                    for j in range(i + self._frames_in - self._frames_out, i + self._frames_in):
                        # 获取对应帧号
                        frame_name = frame_names[j]
                        name_without_ext = os.path.splitext(frame_name)[0]
                        try:
                            fid = int(name_without_ext)
                        except ValueError:
                            import re
                            numbers = re.findall(r'\d+', name_without_ext)
                            if numbers:
                                fid = int(numbers[-1])
                            else:
                                log.warning(f"Could not parse frame ID from {frame_name}, skipping sequence starting at index {i}")
                                all_annos_available = False
                                break
                        
                        if fid in ball_xyvs:
                            annos.append(ball_xyvs[fid])
                        else:
                            # 如果没有标注，创建一个默认的不可见标注
                            annos.append({
                                'center': Center(x=0.0, y=0.0, is_visible=False),
                                'file_name': frame_name,
                                'frame_path': osp.join(clip_frame_dir, frame_name)
                            })
                    
                    if not all_annos_available:
                        continue
                    
                    seq_list.append({'frames': paths, 'annos': annos, 'match': match, 'clip': clip_name})
                    if i % self._step == 0:
                        clip_seq_list.append({'frames': paths, 'annos': annos, 'match': match, 'clip': clip_name})
                

                clip_disps = []
                # compute displacement between consecutive frames
                # 修改这部分，确保ball_xyvs中有连续的索引
                sorted_indices = sorted([int(k) for k in ball_xyvs.keys() if isinstance(k, (int, str)) and str(k).isdigit()])
                for idx in range(len(sorted_indices)-1):
                    curr_idx = sorted_indices[idx]
                    next_idx = sorted_indices[idx+1]
                    
                    # 检查相邻索引是否连续（对于预测用途，我们可能只关心存在的标注）
                    xy1_info = ball_xyvs[curr_idx]
                    xy2_info = ball_xyvs[next_idx]
                    
                    xy1, visi1 = xy1_info['center'].xy, xy1_info['center'].is_visible
                    xy2, visi2 = xy2_info['center'].xy, xy2_info['center'].is_visible
                    if visi1 and visi2:
                        disp = np.linalg.norm(np.array(xy1)-np.array(xy2))
                        disps.append(disp)
                        clip_disps.append(disp)

                # 更新这部分以处理可能不匹配的帧和标注
                for fid in ball_xyvs.keys():
                    if isinstance(fid, (int, str)) and str(fid).isdigit():
                        # 尝试找到对应的文件名
                        frame_filename = None
                        fid_int = int(fid)
                        for fname in frame_names:
                            name_without_ext = os.path.splitext(fname)[0]
                            try:
                                frame_fid = int(name_without_ext)
                            except ValueError:
                                import re
                                numbers = re.findall(r'\d+', name_without_ext)
                                if numbers:
                                    frame_fid = int(numbers[-1])
                                else:
                                    continue
                            
                            if frame_fid == fid_int:
                                frame_filename = fname
                                break
                        
                        if frame_filename:
                            path = osp.join(clip_frame_dir, frame_filename)
                            clip_seq_gt_dict[path] = ball_xyvs[fid]['center']

                clip_seq_list_dict[(match, clip_name)]    = clip_seq_list
                clip_seq_gt_dict_dict[(match, clip_name)] = clip_seq_gt_dict
                clip_seq_disps[(match, clip_name)]         = clip_disps

        return { 'seq_list': seq_list, 
                 'clip_seq_list_dict': clip_seq_list_dict, 
                 'clip_seq_gt_dict_dict': clip_seq_gt_dict_dict,
                 'clip_seq_disps': clip_seq_disps,
                 'num_frames': num_frames, 
                 'num_frames_with_gt': num_frames_with_gt, 
                 'num_matches': num_matches, 
                 'num_rallies': num_rallies,
                 'disp_mean': np.mean(np.array(disps)),
                 'disp_std': np.std(np.array(disps))}

    @property
    def train(self):
        return self._train_all

    @property
    def test(self):
        return self._test_all

    @property
    def train_clips(self):
        return self._train_clips

    @property
    def train_clip_gts(self):
        return self._train_clip_gts

    @property
    def test_clips(self):
        return self._test_clips

    @property
    def test_clip_gts(self):
        return self._test_clip_gts


import os
import os.path as osp
import pickle
import json
import sys
import time
import mmcv
project_root = os.path.abspath ( os.path.join ( os.path.dirname ( __file__ ), '..', '..' ) )
if __name__ == '__main__':
    if project_root not in sys.path:
        sys.path.append ( project_root )
import coloredlogs, logging

logger = logging.getLogger ( __name__ )
coloredlogs.install ( level='DEBUG', logger=logger )

from src.models.model_config import model_cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from src.m_utils.base_dataset import BaseDataset, PreprocessedDataset
from src.models.estimate3d import MultiEstimator
from src.m_utils.evaluate import numpify
from src.m_utils.mem_dataset import MemDataset
from src.models.triangulation import triangulateLinearEigen
from tracker import person_3D_tracker

def export(model, loader, is_info_dicts=False, show=False):
    pose_list = list ()
    for img_id, imgs in enumerate ( tqdm ( loader ) ):
        try:
            pass
        except Exception as e:
            pass
            # poses3d = model.estimate3d ( img_id=img_id, show=False )

        this_imgs = list ()
        for img_batch in imgs:
            this_imgs.append ( img_batch.squeeze ().numpy () )
        ## match and 3d reconstruct
        poses3d = model.predict ( imgs=this_imgs, camera_parameter=camera_parameter, template_name='Unified',
                                      show=show, plt_id=img_id )

        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        root_dir = os.path.abspath(os.path.join(model_dir, '..', '..'))
        ##save resutls
        out_path = os.path.join(root_dir, 'results',f'{img_id:06d}.json')
        mmcv.dump(poses3d,out_path)

        ## tracking
        if img_id==0:
            tracker=person_3D_tracker(poses3d,img_id)
            result=tracker.transform_target(tracker.new_targets)
            resultpath=os.path.join(root_dir,'trackresult',f'final{img_id:06d}.json')
            mmcv.dump(result,resultpath)

            continue
        tracker.update(poses3d,img_id)
        result=tracker.Targettracking()
        resultpath=os.path.join(root_dir,'trackresult',f'final{img_id:06d}.json')
        mmcv.dump(result,resultpath)


        pose_list.append ( poses3d )
    return pose_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument ( '-d', nargs='+', dest='datasets', required=True,
                          choices=['Shelf', 'Campus', 'ultimatum1','Mytest','RealOR'] )
    parser.add_argument ( '-dumped', nargs='+', dest='dumped_dir', default=None )
    #parser.add_argument('--outpath', type=str, help='result path')

    args = parser.parse_args ()

    test_model = MultiEstimator ( cfg=model_cfg )
    for dataset_idx, dataset_name in enumerate ( args.datasets ):
        model_cfg.testing_on = dataset_name

        if dataset_name == 'Mytest':
            dataset_path = model_cfg.mytest_path
            test_range=range(1)
            gt_path = dataset_path

        elif dataset_name == 'RealOR':
            dataset_path = model_cfg.realor_path
            test_range=range(300)
            gt_path = dataset_path

        else:
            logger.error ( f"Unknown datasets name: {dataset_name}" )
            exit ( -1 )

        # read the camera parameter of this dataset
        with open ( osp.join ( dataset_path, 'camera_parameter.pickle' ),
                    'rb' ) as f:
            camera_parameter = pickle.load ( f )

        # using preprocessed 2D poses or using CPN to predict 2D pose
        if args.dumped_dir:
            test_dataset = PreprocessedDataset ( args.dumped_dir[dataset_idx] )
            logger.info ( f"Using pre-processed datasets {args.dumped_dir[dataset_idx]} for quicker evaluation" )
        else:

            test_dataset = BaseDataset ( dataset_path, test_range )

        test_loader = DataLoader ( test_dataset, batch_size=1, pin_memory=True, num_workers=6, shuffle=False )
        pose_in_range = export ( test_model, test_loader, is_info_dicts=bool ( args.dumped_dir ), show=True )




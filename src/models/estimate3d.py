
import sys
import os
import os.path as osp
import torch
# Config project if not exist
project_path = osp.abspath ( osp.join ( osp.dirname ( __file__ ), '..', '..' ) )
if project_path not in sys.path:
    sys.path.insert ( 0, project_path )

import psutil

from src.models.model_config import model_cfg
from backend.my2dest import Estimator_2d
import time
import coloredlogs, logging

logger = logging.getLogger ( __name__ )
coloredlogs.install ( level='DEBUG', logger=logger )
import os.path  as osp
import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.m_utils.geometry import geometry_affinity, get_min_reprojection_error, check_bone_length, bundle_adjustment, \
    multiTriIter
from backend.CamStyle.feature_extract import FeatureExtractor
from src.m_utils.mem_dataset import MemDataset
from src.models.matchSVT import matchSVT
from src.m_utils.visualize import show_panel_mem, plotPaperRows
# from src.models import pictorial
from src.m_lib import pictorial
#import triangulation

class MultiEstimator ( object ):
    def __init__(self, cfg, debug=False):
        self.est2d = Estimator_2d ( DEBUGGING=debug )
        #self.extractor = FeatureExtractor ()
        self.cfg = cfg
        self.dataset = None
        self.match_dic=[{},{},{},{},{}]
   



    def predict(self, imgs, camera_parameter, template_name='Shelf', show=False, plt_id=0):
        info_dict = self._infer_single2d ( imgs,img_id=plt_id )
        self.dataset = MemDataset ( info_dict=info_dict, camera_parameter=camera_parameter,
                                    template_name=template_name )
        return self._estimate3d ( plt_id, show=show, plt_id=plt_id )

    def _infer_single2d(self, imgs, img_id=0):
        info_dict = dict ()
        for cam_id, img in enumerate ( imgs ):
            ## load 2d results
            results = self.est2d.get_2d ( cam_id, img_id )
            this_info_dict = {'image_data': cv2.cvtColor ( img.copy (), cv2.COLOR_BGR2RGB )}
            this_info_dict[img_id] = list ()
            for person_id, result in enumerate ( results ):
                this_info_dict[img_id].append ( dict () )
                this_info_dict[img_id][person_id]['pose2d'] = np.array(result['keypoints']).flatten()
                this_info_dict[img_id][person_id]['identity'] = result['track_id']+1
                this_info_dict[img_id][person_id]['cam_id'] = cam_id

                # NOTE: bbox is (x, y) (W, H) format where x and y is up-left point.
                [x,y,w,h]=result['bbox'][:-1]
                x=x-w/2
                y=y+h/2
                this_info_dict[img_id][person_id]['bbox'] =[x,y,w,h]
                
                bb = np.array ( [x,y,w,h], dtype=int )
                xs=[bb[0],bb[0] + bb[2]]
                ys=[bb[1],bb[1] + bb[3]]
                xs=np.maximum(xs,0)
                ys=np.maximum(ys,0)
                xs=np.minimum(xs,1280-1)
                ys=np.minimum(ys,720-1)
                if  ys[0]==ys[1] or xs[0]==xs[1]:
                    ys[1]+=1
                    xs[1]+=1

                                   
                cropped_img = img[ ys[0]:ys[1], xs[0]:xs[1]]
                # numpy format of crop idx is changed to json
                this_info_dict[img_id][person_id]['cropped_img'] = cv2.cvtColor ( cropped_img.copy (),cv2.COLOR_BGR2RGB )

            info_dict[cam_id] = this_info_dict
        return info_dict

    def _estimate3d(self, img_id, show=False, plt_id=0):
        data_batch = self.dataset[img_id]

        dimGroup = self.dataset.dimGroup[img_id]

        info_list = list ()
        for cam_id in self.dataset.cam_names:
            info_list += self.dataset.info_dict[cam_id][img_id]
        num_detect=len(info_list)
        affinity2d=np.zeros((num_detect,num_detect),dtype=int)
        if img_id>0:
            for i in range(num_detect):
                for j in range(num_detect):
                    camidi=info_list[i]['cam_id']
                    camidj=info_list[j]['cam_id']
                    trackidi=info_list[i]['identity']
                    trackidj = info_list[j]['identity']
                    x=self.match_dic[camidi]
                    if  trackidi in x and x[trackidi][camidj] ==trackidj:
                        if trackidj!=0:
                            affinity2d[i][j]=1.0

        affinity2d_mat=torch.tensor(affinity2d)



        #print(np.array ( [i['pose2d'] for i in info_list] ))
        pose_mat = np.array ( [i['pose2d'] for i in info_list] ).reshape ( -1, model_cfg.joint_num, 3 )[..., :2]
        geo_affinity_mat = geometry_affinity ( pose_mat.copy (), self.dataset.F.numpy (),
                                               self.dataset.dimGroup[img_id] )
        W = torch.tensor ( geo_affinity_mat )
        # if self.cfg.metric == 'geometry mean':
        #     W = torch.sqrt ( affinity_mat * geo_affinity_mat )
        # elif self.cfg.metric == 'circle':
        #     W = torch.sqrt ( (affinity_mat ** 2 + geo_affinity_mat ** 2) / 2 )
        # elif self.cfg.metric == 'Geometry only':
        #     W = torch.tensor ( geo_affinity_mat )
        # elif self.cfg.metric == 'ReID only':
        #     W = torch.tensor ( affinity_mat )
        # else:
        #     logger.critical ( 'Get into default option, are you intend to do it?' )
        #     _alpha = 0.1
        #     W = _alpha * affinity_mat + (1 - _alpha) * geo_affinity_mat
        #print(W)
        _alpha=0.15
        W = _alpha * affinity2d_mat + (1 - _alpha) * geo_affinity_mat
        W = torch.sqrt((affinity2d_mat ** 2 + geo_affinity_mat ** 2) / 2)
        W[torch.isnan ( W )] = 0  # Some times (Shelf 452th img eg.) torch.sqrt will return nan if its too small
        sub_imgid2cam = np.zeros ( pose_mat.shape[0], dtype=np.int32 )
        for idx, i in enumerate ( range ( len ( dimGroup ) - 1 ) ):
            sub_imgid2cam[dimGroup[i]:dimGroup[i + 1]] = idx

        num_person = 10
        X0 = torch.rand ( W.shape[0], num_person )

        # Use spectral method to initialize assignment matrix.
        if self.cfg.spectral:
            eig_value, eig_vector = W.eig ( eigenvectors=True )
            _, eig_idx = torch.sort ( eig_value[:, 0], descending=True )

            if W.shape[1] >= num_person:
                X0 = eig_vector[eig_idx[:num_person]].t ()
            else:
                X0[:, :W.shape[1]] = eig_vector.t ()
        ##solve matching matrix
        match_mat = matchSVT ( W, dimGroup, alpha=self.cfg.alpha_SVT, _lambda=self.cfg.lambda_SVT,
                               dual_stochastic_SVT=self.cfg.dual_stochastic_SVT )


        bin_match = match_mat[:, torch.nonzero ( torch.sum ( match_mat, dim=0 ) > 1.9 ).squeeze ()] > 0.9
        bin_match = bin_match.reshape ( W.shape[0], -1 )

        matched_list = [[] for i in range ( bin_match.shape[1] )]
        for sub_imgid, row in enumerate ( bin_match ):
            if row.sum () != 0:
                row=row*1
                #print(row)
                pid = row.argmax ()
                matched_list[pid].append ( sub_imgid )

        matched_list = [np.array ( i ) for i in matched_list]

        mathresult=self.findmath(matched_list,sub_imgid2cam,pose_mat,img_id)
        #print(mathresult)
        posedict=self.my_pose_withmatch(mathresult,img_id)
        #return multi_pose3d
        return posedict


    
    def findmath(self,matchlist,sub_imgid2cam,pose_mat,img_id=0):
    ## original matchlist using sub_image_id, this function change sub_image_id to camera_id+track_id
    ##input matchlist
    ##output [id,id..],[id,id..] list, each element represents a person with track id from 5 cameras
        matchresult=[]
        for i,person in enumerate(matchlist):
            if len(person)==1:
                continue
            onematch=np.zeros(len(self.dataset.P),dtype=int)
            for j,view in enumerate(person):
                cam_id=sub_imgid2cam[view]
                pose2d=pose_mat[view]
                gtposes=self.dataset.info_dict[cam_id][img_id]
                for personid,gtpose in enumerate(gtposes):
                    x1=gtpose['pose2d'][0]
                    y1=gtpose['pose2d'][1]
                    check1=gtpose['pose2d'][3]
                    #print(x1,y1)
                    x2=pose2d[0][0]
                    y2=pose2d[0][1]
                    check2=pose2d[1][0]
                    #print(x2,y2)
                    if x1==x2 and y1==y2 and check1==check2:
                        onematch[cam_id]=gtpose['identity']
            for view_id in range(len(onematch)):
                identity2d=onematch[view_id]
                self.match_dic[view_id][identity2d]=onematch
            matchresult.append(onematch.tolist())
        
        return matchresult
    
    def my_pose_withmatch(self,matchlist,img_id):
        ## generate 3d pose according to matchlist

        ##input matchlist [id,id..],[id,id,...]...
        ##output list of 3d pose

        pose_person=[]
        for i,person in enumerate(matchlist):
            pose_views=[]
            ##load 2d results
            for cam_id in range(5):
                if person[cam_id]==0:
                    pose_views.append(np.zeros((9,3)))
                else:
                    results = self.est2d.get_2d(cam_id, img_id)
                    for person_id, result in enumerate(results):
                        if result['track_id']==(person[cam_id]-1):
                            pose_cam=np.array(result['keypoints'])

                    if len(pose_cam)!=9:
                        pose_views.append(np.zeros((9, 3)))
                    else:
                        pose_views.append(pose_cam)
            ##reconstruct
            pose_trans=np.array(pose_views).transpose(1,0,2)
            joints_3d=[]
            for joint_id,joint in enumerate(pose_trans):
                Projections=self.dataset.P
                #order according to confidence
                order=(-joint[:,2]).argsort()
                sortjoint=joint[order]
                sortcam=Projections[order]
                #use first three views
                limitjoint = np.array([x[:-1] for i, x in enumerate(sortjoint[0:3]) if x[2] > 0.1],dtype=float)
                limitcam = [sortcam[i] for i, x in enumerate(sortjoint[0:3]) if x[2] > 0.1]

                if len(limitjoint)< 2:
                    limitjoint = np.array([x[:-1] for  x in sortjoint[0:2] ], dtype=float)
                    limitcam = [sortcam[i] for i, x in enumerate(sortjoint[0:2]) ]


                joint_3d=self.triangulateLinearEigen(limitcam,limitjoint)
                joints_3d.append(joint_3d)

            person_dict={'matchlist':person,'pose3d':joints_3d}
            pose_person.append(person_dict)

        return  pose_person


    def triangulateLinearEigen(self,cameras, x):
        '''
        Triangulate using the linear eigen method
        cameras: list of cameras, shape [N,]
        x: image projection coordinates, shape [N,2] or [N,2,1]
        '''
        longX = True
        if len(x.shape) == 2:
            longX = False
            x = x[:,:,None] # Ensure shape [N,2,1]
        x = np.concatenate([x, np.ones(shape=[x.shape[0],1,1])], axis=1) # [N,3,1]
    
        P = np.asarray([cam for cam in cameras]) # [N,3,4]
        A1 = x[:,0,:]*P[:,2,:] - P[:,0,:] # [N,1]*[N,4] = [N,4]
        A2 = x[:,1,:]*P[:,2,:] - P[:,1,:] # [N,1]*[N,4] = [N,4]
        A = np.concatenate([A1, A2], axis=0) # [2N,4]
    
        X,_,_,_ = np.linalg.lstsq(A[:,:3], -A[:,3,None]) # [2N,3]\[2N,1] = [3,1]

        return X if longX else X[:,0]
    



if __name__ == '__main__':
    import pickle
    import scipy.io as scio
    from src.models.model_config import model_cfg
    from glob import glob
    from tqdm import tqdm
    from src.m_utils.base_dataset import BaseDataset
    from torch.utils.data import DataLoader, Subset
    import random

    est = MultiEstimator ( model_cfg, debug=False )
    with open ( osp.join ( model_cfg.shelf_path, 'camera_parameter.pickle' ), 'rb' ) as f:
        test_camera_parameter = pickle.load ( f )
    test_dataset = BaseDataset ( model_cfg.shelf_path, range ( 300, 600 ) )
    test_dataset = Subset ( test_dataset, random.sample ( range ( 300 ), 50 ) )
    test_loader = DataLoader ( test_dataset, batch_size=1, pin_memory=True, num_workers=12, shuffle=False )
    for imgs in tqdm ( test_loader ):
        this_imgs = list ()
        for img_batch in imgs:
            this_imgs.append ( img_batch.squeeze ().numpy () )
        poses3d = est.predict ( imgs=this_imgs, camera_parameter=test_camera_parameter, show=False,
                                template_name='Shelf' )
        # print ( poses3d )

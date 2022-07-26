import numpy as np
import mmcv
import os
from munkres import Munkres, make_cost_matrix

class Target():

    def __init__(self,pose,track_2d,Identify,frame_id):
        self.pose=np.array(pose,dtype=float)      #3Dpose 9*3 joints*xyz
        self.track_2d=track_2d  #list of 2d track id [id,id,id,id,id]
        self.Identify=Identify  # id of the target
        self.frame_id=frame_id


    def update(self,pose,track_2d,frame_id):
        self.pose=pose
        self.track_2d=track_2d
        self.frame_id=frame_id


class person_3D_tracker():
    def __init__(self,people,frame_id):
        ##input people:load from match result for each frame [id pose, id pose,....]

        self.num_person=len(people)
        targets=[]
        for i,person in enumerate(people):
            target=Target(person['pose3d'],person['matchlist'],i,frame_id)
            targets.append(target)
        self.new_targets=targets

        self.num_id=self.num_person
        self.threshold=100

    def update(self,people,frame_id):
        self.num_person=len(people)
        targets=[]
        for i,person in enumerate(people):
            target=Target(person['pose3d'],person['matchlist'],-1,frame_id)
            targets.append(target)
        self.old_targets=self.new_targets
        self.new_targets=targets

    def Targettracking(self):
        ##track new targets based on old targets


        frame_id=self.new_targets[0].frame_id

        newTs=self.new_targets
        oldTs=self.old_targets
        M = len(self.new_targets)
        N = len(self.old_targets)
        AffinityM = np.zeros((N, M),dtype=float)  #affinity matrix
        old_exit=[]
        ##compute affinity matrix
        for j, oldT in enumerate(oldTs):
            for i, newT in enumerate(newTs):
                AffinityM[j][i] = self.affinity(newT, oldT)



        print(AffinityM)
        ##Kuhn–Munkres algorithm to find correct old-new pairs
        cost_matrix = make_cost_matrix(AffinityM, lambda x: 1 - x)
        mk = Munkres()
        index = mk.compute(cost_matrix.copy())

        newthrH=0
        #update identity
        for pair in index:
            self.new_targets[pair[1]].Identify=self.old_targets[pair[0]].Identify
            self.old_targets[pair[0]].frame_id=frame_id
            newthrH+=self.compute_threshold(self.new_targets[pair[1]],self.old_targets[pair[0]])

        ## update threshold based on pair
        #self.threshold=newthrH/len(index)
        #print(self.threshold)
        ##add unmatch target into unmath list
        #self.unmath=[]


        for target in self.old_targets:
            if frame_id-target.frame_id==1:
                self.new_targets.append(target)



        ##add new target into list
        for target in self.new_targets:
            if target.Identify==-1:
                target.Identify=self.num_id
                self.num_id+=1

        return self.transform_target(self.new_targets)


    def compute_threshold(self,T1,T2):
        ## automatically compute the threshold

        pose1 = T1.pose
        pose2 = T2.pose
        simi = 0
        ##only compair torse head stomach
        for i in [-1, -2, -3]:
            dist = np.linalg.norm(pose1[i] - pose2[i])
            simi = simi + dist
        return  simi/3


    def transform_target(self,targets):
        ##transform result into dictional format
        dirlist=[]
        for target in targets:
            person_dir={'track_id':target.Identify,'keypoints':target.pose}
            dirlist.append(person_dir)

        return  dirlist



    def affinity(self,T1,T2):
        ##compute affinity score between two targets
        ##imput  one old target and one new target
        ##output affinity score

        ##weight
        w1 = 0.6
        w2 = 0.4
        bias = 0.5

        # A1 3D affinity score
        A1 = self.Similarity_3d(T1, T2)

        # A2 2d track score
        a = T1.track_2d
        b = T2.track_2d
        count = 0
        correct = 0
        for i in range(5):
            if a[i] == 0 or b[i] == 0:
                continue
            if a[i] == b[i]:
                correct = correct + 1
            count = count + 1

        A2 = correct / (count + bias)

        A = w1 * A1 + w2 * A2
        return A

    def Similarity_3d(self,T1, T2):
        ##compute 3d similarity


        pose1 = T1.pose
        pose2 = T2.pose
        simi = 0
        ##only compair torse head stomach
        for i in [-1, -2, -3]:
            dist = np.linalg.norm(pose1[i] - pose2[i])
            simi = simi + (1 - dist / self.threshold)

        return simi / 3

if __name__ == '__main__':
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    for frame_id in range(300):



        filepath = os.path.join(model_dir, 'results', f'{frame_id:06d}.json')
        targets = mmcv.load(filepath)
        if frame_id==0:
            tracker=person_3D_tracker(targets,frame_id)
            result=tracker.transform_target(tracker.new_targets)
            resultpath=os.path.join(model_dir,'trackresult',f'final{frame_id:06d}.json')
            mmcv.dump(result,resultpath)

            continue

        tracker.update(targets,frame_id)
        result=tracker.Targettracking()
        #print(result)
        resultpath=os.path.join(model_dir,'trackresult',f'final{frame_id:06d}.json')
        mmcv.dump(result,resultpath)



import mmcv
import os
import os.path as osp
import numpy as np
class Estimator_2d (  ):

    def __init__(self, DEBUGGING=False):
        self.bbox_detector =DEBUGGING

        #      result1={'keypoints':[],'bbox':[]}
        result2 = {
            'keypoints': [1042, 370, 0, 1077, 372, 1, 1083, 307, 1, 1012, 297, 0, 982, 353, 0, 964, 360, 0, 1046, 278,
                          1, 1053, 230, 1, 1024, 371, 1
                          ], 'bbox': [947.48, 217.6, 148.05, 173.35]}
        result3 = {
            'keypoints': [548, 340, 0, 618, 400, 0, 611, 337, 1, 510, 341, 1, 475, 416, 1, 523, 367, 0, 560, 327, 1,
                          570, 233, 1, 557, 476, 1
                          ], 'bbox': [467.71, 211.38, 177.49, 270.85]}
        result4 = {
            'keypoints': [655, 298, 1, 605, 282, 1, 620, 234, 1, 693, 238, 1, 699, 288, 0, 677, 301, 1, 664, 218, 1,
                          670, 165, 1, 659, 300, 1
                          ], 'bbox': [588.85, 153.41, 130.7, 179.05]}
        result5 = {
            'keypoints': [714, 208, 1, 733, 280, 1, 735, 344, 0, 655, 390, 1, 641, 437, 1, 629, 413, 0, 700, 366, 1,
                          715, 307, 1, 690, 465, 1
                          ], 'bbox': [616.06, 188.99, 136.33, 275.01]}
        results1 = [result2, result3, result4, result5]

        result1 = {
            'keypoints': [436, 507, 1, 498, 593, 1, 561, 513, 1, 632, 431, 1, 586, 476, 0, 496, 465, 1, 593, 446, 1,
                          599, 328, 1, 550, 603, 1
                          ], 'bbox': [417.96, 270.73, 236.33, 354.14]}
        result2 = {
            'keypoints': [753, 224, 0, 767, 222, 0, 771, 191, 1, 752, 191, 1, 745, 227, 1, 727, 230, 0, 762, 182, 1,
                          764, 153, 1, 757, 228, 1
                          ], 'bbox': [709.68, 137.77, 84.631, 115.57]}
        result3 = {
            'keypoints': [596, 255, 1, 627, 282, 0, 650, 244, 1, 626, 255, 1, 601, 313, 1, 584, 268, 1, 639, 244, 1,
                          652, 178, 1, 595, 335, 0
                          ], 'bbox': [568.45, 163.84, 104.22, 191.3]}
        result4 = {
            'keypoints': [571, 208, 1, 542, 202, 1, 541, 160, 1, 579, 158, 1, 578, 196, 0, 583, 205, 1, 572, 145, 1,
                          583, 108, 1, 558, 217, 1
                          ], 'bbox': [520.39, 93.222, 88.494, 146.57]}
        result5 = {
            'keypoints': [681, 153, 1, 675, 188, 1, 681, 244, 0, 662, 271, 0, 647, 301, 0, 634, 276, 0, 680, 251, 1,
                          692, 221, 1, 659, 305, 0
                          ], 'bbox': [608.48, 136.67, 106.14, 188.35]}
        results2 = [result1, result2, result3, result4, result5]

        result1 = {
            'keypoints': [961, 423, 1, 971, 390, 1, 1006, 323, 1, 1154, 387, 1, 1109, 472, 1, 1010, 469, 1, 1076, 330,
                          1, 1058, 239, 1, 1022, 491, 1
                          ], 'bbox': [928.12, 196.5, 196.34, 347.7]}
        result2 = {
            'keypoints': [680, 199, 0, 685, 175, 0, 684, 153, 0, 686, 161, 0, 688, 188, 0, 677, 203, 0, 679, 145, 0,
                          663, 124, 1, 689, 196, 0
                          ], 'bbox': [640.28, 99.934, 68.406, 121.17]}
        result3 = {
            'keypoints': [704, 271, 1, 729, 267, 0, 728, 223, 0, 749, 248, 1, 757, 308, 1, 708, 294, 1, 738, 229, 1,
                          726, 176, 1, 743, 358, 1
                          ], 'bbox': [676.05, 155.97, 96.396, 200.86]}
        result4 = {
            'keypoints': [599, 264, 1, 576, 283, 1, 558, 240, 1, 573, 221, 0, 585, 252, 0, 595, 259, 0, 566, 214, 1,
                          564, 177, 1, 580, 288, 1
                          ], 'bbox': [528.72, 151.14, 83.991, 157.92]}
        # result5={'keypoints':[],'bbox':[653.62, 129.22 ,103.39, 177.43]}
        results3 = [result1, result2, result3, result4]

        result1={'keypoints':[1191,468,1,1248,469,1,1269,414,1,1132,371,1,1100,413,1,1123,439,1,1205,374,1,1223,317,1,1150,476,1],'bbox':[1101.5 ,268.65 ,172.99 ,265.89]}
        result2={'keypoints':[662,231,0,682,218,0,688,182,0,700,191,1,703,225,0,689,231,0,685,177,1,681,165,1,687,227,0],'bbox':[654.77, 141.1, 70.624, 114.08]}
        result3={'keypoints':[813,307,1,796,312,1,829,270,1,875,294,1,865,341,1,839,326,1,853,273,1,854,229,1,836,353,1],'bbox':[787.68, 211.21, 111.61, 174.62]}
        result4={'keypoints':[684,292,0,675,302,1,666,262,1,650,242,0,647,275,0,670,286,0,663,234,1,674,197,1,652,321,1],'bbox':[630.47, 182.08, 70.41 ,150.84]}
        result5={'keypoints':[784,196,1,776,229,1,787,256,1,825,283,0,819,318,0,789,320,1,810,265,1,817,231,1,795,312,1],'bbox':[754.07, 179.29, 89.725 ,161.34]}
        results4=[result1,result2,result3,result4,result5]

        result1 = {
            'keypoints': [960, 191, 1, 922, 191, 1, 933, 147, 1, 1003, 131, 1, 1011, 168, 1, 1002, 184, 1, 964, 132, 1,
                          967, 84, 1, 970, 210, 1
                          ], 'bbox': [914.48, 32.378, 123.22, 235.15]}
        result2 = {
            'keypoints': [484, 234, 0, 483, 215, 0, 474, 188, 0, 498, 187, 1, 518, 216, 1, 497, 238, 0, 477, 183, 1,
                          461, 162, 1, 495, 229, 1
                          ], 'bbox': [436.96, 142.93, 97.772, 112.79]}
        result3 = {
            'keypoints': [708, 225, 1, 686, 237, 1, 694, 189, 1, 746, 185, 1, 758, 225, 1, 726, 219, 1, 719, 179, 1,
                          710, 139, 1, 725, 246, 1
                          ], 'bbox': [665.72, 115.76, 113.78, 167.07]}
        result4 = {
            'keypoints': [572, 283, 0, 602, 287, 1, 575, 240, 1, 531, 240, 1, 541, 279, 0, 561, 280, 0, 553, 221, 1,
                          548, 185, 1, 566, 306, 1
                          ], 'bbox': [509.31, 159.71, 110.92, 162.74]}
        result5 = {
            'keypoints': [621, 149, 1, 622, 184, 1, 642, 199, 1, 682, 204, 0, 690, 235, 0, 673, 250, 1, 660, 195, 1,
                          660, 163, 1, 660, 246, 1
                          ], 'bbox': [599.29, 121.14, 111.67, 146.76]}

        results5=[result1,result2,result3,result4,result5]
        self.matrix=[results1,results2,results3,results4,results5]

    def estimate_2d(self, img, img_id,cam_id):

        dump_results=self.matrix[cam_id]
        return dump_results
        
    def get_2d(self,cam_id,frame_id):
        ##load 2d results from files

        cam=cam_id+1
        pathPos="datasets/realor/poses/cam"+str(cam)
        model_dir = os.path.abspath ( os.path.join ( os.path.dirname ( __file__ ) ) )
        root_dir = os.path.abspath ( os.path.join ( model_dir, '..' ) )        
        path=osp.join(root_dir,pathPos)
        img = osp.join(path, f'{frame_id:06d}.json')
        persons=mmcv.load(img)
        persons=self.deduplication(persons)


    
    
        
        return persons

    def deduplication(self,persons):
        # remove same pose
        
        N = len(persons)
        newperson=[]
        check_dic={}
        ioumat=np.zeros((N,N),dtype=float)
        for i in range(N):
            if (i in check_dic)==True:
                continue

            for j in range(i+1, N):
                boxA = persons[i]['bbox']
                boxB = persons[j]['bbox']
                iou=self.IoU(self.bboxformtran(boxA) ,self.bboxformtran(boxB) )
                ioumat[i][j]=iou
                ##if two detections are overlapping
                if iou> 0.2:

                    areaA=boxA[2]*boxA[3]
                    areaB = boxB[2] * boxB[3]
                    pose1 = np.array(persons[i]['keypoints'], dtype=float).transpose(1, 0)[:-1]
                    pose2 = np.array(persons[j]['keypoints'], dtype=float).transpose(1, 0)[:-1]
                    pose1 = pose1.transpose(1, 0)
                    pose2 = pose2.transpose(1, 0)
                    simi = 0
                    threshold=2
                    #compute similarity, count how many points are the same

                    for joint in range(9):
                        dist = np.linalg.norm(pose1[joint] - pose2[joint])
                        #print(dist)
                        if dist<threshold:
                            simi+=1
                    if simi>1:
                        if areaA>areaB:
                            check_dic[j]=-1
                        else:
                            check_dic[i]=-1

            if (i in check_dic)==False:
                newperson.append(persons[i])
        return newperson
    def bboxformtran(self,bbox):
    #change the format of bounding boxes
        x,y,w,h=bbox[:-1]
        x1=x
        y1=y
        x2=x+w
        y2=y+h
        return [x1,y1,x2,y2]

    def IoU(self,boxA,boxB):
    ## compute IOU

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        #print(interArea)
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        #print(iou)
        return iou

x=Estimator_2d()
x.get_2d(3,5)



import os
import numpy as np
import cv2
from scipy.optimize import least_squares
import plotly.graph_objects as go
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

scale = 0.7
class VisualOdometry():
    def __init__(self, data_dir):
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(data_dir + '/calib.txt')
        self.begin_pose = self._load_poses(data_dir + '/begin_pose.txt')

        block = 11
        P1 = block**2 * 8
        P2 = block**2 * 16
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)

        self.prev_disp = None
        self.cur_disp = None
        self.kp_full = None
        self.des_full = None
        self.kp2 = None
        self.des2 = None

        self.orb = cv2.ORB_create(2000)
        
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        self.lk_params = dict(winSize=(15, 15), flags=cv2.MOTION_AFFINE, maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            P_l = np.reshape(np.fromstring(f.readline(), dtype=np.float64, sep=' '), (3, 4))
            P_l[0, :] *= scale
            P_l[1, :] *= scale

            K_l = P_l[0:3, 0:3]
          
            P_r = np.reshape(np.fromstring(f.readline(), dtype=np.float64, sep=' '), (3, 4))
            P_r[0, :] *= scale
            P_r[1, :] *= scale
            K_r = P_r[0:3, 0:3]

        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_poses(filepath):
        pose = []
        with open(filepath, 'r') as f:
            pose = np.vstack((np.fromstring(f.readline(), dtype=np.float64, sep=' ').reshape(3, 4), [0, 0, 0, 1]))
        return pose


    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        r = dof[:3]
        R, _ = cv2.Rodrigues(r)
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = self.P_l @ transf
        b_projection = self.P_l @ np.linalg.inv(transf)

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def get_tiled_keypoints_orb(self, img, kp_full, des_full, tile_h, tile_w):
        """
        Splits the image into tiles and detects the best keypoints in each tile using ORB

        Parameters
        ----------
        img (ndarray): The image to find keypoints in. Shape (height, width)
        tile_h (int): The tile height
        tile_w (int): The tile width

        Returns
        -------
        kp_list (list): A list of all keypoints
        descriptors (list): A list of descriptors for each keypoint
        """
        h, w = img.shape
        all_keypoints = []
        all_descriptors = []
        
        #kp_full, des_full = self.orb.detectAndCompute(img, None)
        
        if kp_full is None:
            return [], []
        
        tile_keypoints = {}
        tile_descriptors = {}
        
        for i, kp in enumerate(kp_full):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            tile_x = x // tile_w
            tile_y = y // tile_h
            
            if tile_y >= h // tile_h or tile_x >= w // tile_w:
                continue
                
            key = (tile_y, tile_x)
            if key not in tile_keypoints:
                tile_keypoints[key] = []
                tile_descriptors[key] = []
            
            tile_keypoints[key].append(kp)
            if des_full is not None:
                tile_descriptors[key].append(des_full[i])
        
        for key in tile_keypoints:
            kps = tile_keypoints[key]
            descs = tile_descriptors.get(key, [])
            
            if len(kps) ==0:
                sorted_idx = np.argsort([kp.response for kp in kps])[::-1][:15]
                kps = [kps[i] for i in sorted_idx]
                if descs:
                    descs = [descs[i] for i in sorted_idx]
            
            all_keypoints.extend(kps)
            all_descriptors.extend(descs)
        
        return all_keypoints, all_descriptors

    def track_keypoints_orb(self, img1, img2, kp2, des2, kp1, des1):
        """
        Tracks the keypoints between frames using feature matching

        Parameters
        ----------
        img1 (ndarray): i-1'th image
        img2 (ndarray): i'th image
        kp1 (list): Keypoints in i-1'th image
        des1 (ndarray): Descriptors for keypoints in i-1'th image

        Returns
        -------
        trackpoints1 (ndarray): Tracked keypoints for i-1'th image
        trackpoints2 (ndarray): Tracked keypoints for i'th image
        """
        if des1 is None or len(kp1) == 0:
            return np.array([]), np.array([])
        
        #kp2, des2 = self.orb.detectAndCompute(img2, None)
        
        if des2 is None or len(kp2) == 0:
            return np.array([]), np.array([])
        
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        good_matches = []
        try:
            good_matches = np.array([m for m, n in matches if m.distance < 0.7 * n.distance])
        except ValueError:
            pass
        
        if len(good_matches) == 0:
            return np.array([]), np.array([])

        dist = [x.distance for x in good_matches]
        if len(good_matches)>500:
            good_matches = good_matches[np.argsort(dist)][:500]
        
        draw_params = dict(matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_DEFAULT)

        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        img3 = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good_matches, None, **draw_params)
        cv2.imshow("ORB Matches", img3)
        cv2.waitKey(1)
        
        trackpoints1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        trackpoints2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
       
        trackpoints1_flow = np.expand_dims(trackpoints1, axis=1)
        trackpoints2_flow, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1_flow, None, **self.lk_params)
        
        if trackpoints2_flow is not None:
            flow_diff = np.linalg.norm(trackpoints2 - trackpoints2_flow.squeeze(), axis=1)
            flow_valid = flow_diff < 5.0
            
            trackpoints1 = trackpoints1[flow_valid]
            trackpoints2 = trackpoints2[flow_valid]
        
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(np.logical_and(trackpoints2[:, 0] >= 0, trackpoints2[:, 0] < w),
            np.logical_and(trackpoints2[:, 1] >= 0, trackpoints2[:, 1] < h)), True, False)
        
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]
        
        return trackpoints1, trackpoints2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            valid_idx = np.where(np.logical_and(np.logical_and(q_idx[:, 0] >= 0, q_idx[:, 0] < disp.shape[1]),
                np.logical_and(q_idx[:, 1] >= 0, q_idx[:, 1] < disp.shape[0])))[0]
            
            if len(valid_idx) == 0:
                return np.array([]), np.array([])
            
            q_idx_valid = q_idx[valid_idx]
            disp_values = disp[q_idx_valid[:, 1], q_idx_valid[:, 0]]
            mask = np.logical_and(min_disp < disp_values, disp_values < max_disp)
            
            return valid_idx[mask], disp_values[mask]
        
        # Get the disparity's for the feature points and mask for min_disp & max_disp
        valid1, disp1_values = get_idxs(q1, disp1)
        valid2, disp2_values = get_idxs(q2, disp2)
        
        if not(len(valid1) and len(valid2)):
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        valid_intersection = np.intersect1d(valid1, valid2)
        
        if not(len(valid_intersection)):
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        
        q1_l = q1[valid_intersection]
        q2_l = q2[valid_intersection]
        
        # Calculate the right feature points 
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1_values[[np.where(valid1 == v)[0][0] for v in valid_intersection]]
        q2_r[:, 0] -= disp2_values[[np.where(valid2 == v)[0][0] for v in valid_intersection]]
        
        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images 
        
        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        if len(q1_l) == 0:
            return np.array([]),np.array([])
        
        # Triangulate points from i-1'th image
        try:
            Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
            Q1 = (Q1[:3] / Q1[3]).T
            
            Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
            Q2 = (Q2[:3] / Q2[3]).T
            
            return Q1, Q2
        except:
            return np.array([]),np.array([])

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        if not(len(q1) and len(q2)):
            return np.eye(4)
        
        early_termination_threshold = 5
        num_points = 6
        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0
        out_pose = np.zeros(6)

        for _ in range(max_iter):
            # Choose 6 random feature points
            if q1.shape[0] < num_points:
                break
                
            sample_idx = np.random.choice(range(q1.shape[0]), num_points, replace=False)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            try:
                opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=50,
                                        args=(sample_q1, sample_q2, sample_Q1, sample_Q2))
            except:
                continue

            # Calculate the error for the optimized transformation
            try:
                error = np.sum(np.linalg.norm(self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2).reshape((Q1.shape[0] * 2, 2)), axis=1))
            except:
                continue

            # Check if the error is less the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
                
            if early_termination == early_termination_threshold:
                break

        if np.all(out_pose == 0):
            return np.eye(4)
            
        r = out_pose[:3]
        R, _ = cv2.Rodrigues(r)
        t = out_pose[3:]
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix

    def get_pose(self, img1_l, img2_l):
        """
        Calculates the transformation matrix for the i'th frame

        Parameters
        ----------
        i (int): Frame index

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """          
        """
        kp2, des2 = self.orb.detectAndCompute(img2_l, None)
        self.cur_disp = self.disparity.compute(img2_l, img2_r).astype(np.float32)/16
        """
        kp1_l, des1_l = self.get_tiled_keypoints_orb(img1_l, self.kp_full, self.des_full, 10, 20)
        
        if len(kp1_l) == 0:
            return np.eye(4)
        
        tp1_l, tp2_l = self.track_keypoints_orb(img1_l, img2_l, self.kp2, self.des2, kp1_l, np.array(des1_l))
        self.kp_full, self.des_full = self.kp2, self.des2

        if not(len(tp1_l) and len(tp2_l)):
            return np.eye(4)
        
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.prev_disp, self.cur_disp)
        self.prev_disp = self.cur_disp

        if not(len(tp1_l) and len(tp2_l) and len(tp1_r) and len(tp2_r)):
            return np.eye(4)
        
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)
        if not(len(Q1) and len(Q2)):
            return np.eye(4)

        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        return transformation_matrix


def stop_criteria(i: int, batch):
    return i < 754-batch

def get_current_images(i: int, batch):
    image_paths1 = [os.path.join("08/image_l", file) for file in sorted(os.listdir("08/image_l"))][i: i+batch]
    image_paths2 = [os.path.join("08/image_r", file) for file in sorted(os.listdir("08/image_r"))][i: i+batch]
    return [cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), None, fx=scale, fy=scale) for path in image_paths1], [cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), None, fx=scale, fy=scale) for path in image_paths2]

def generate_track(data_dir, stop_criteria=stop_criteria, get_current_images=get_current_images):

    proces = 8
    executor = ThreadPoolExecutor(proces)

    VO = [VisualOdometry(data_dir) for i in range(proces)]

    def declare_params(j, current_image_l, current_image_r):
        VO[j].cur_disp = np.divide(VO[j].disparity.compute(current_image_l, current_image_r).astype(np.float32), 16)
        VO[j].kp2, VO[j].des2 = VO[j].orb.detectAndCompute(current_image_l, None)

    cur_pose = None
    i = 0
    
    while stop_criteria(i, proces):
        if i < 1:
            prev_images_l, prev_images_r = get_current_images(0, 1)
            cur_pose = VO[0].begin_pose
            i += 1

        else:
            prev_pose = cur_pose
            current_image_l, current_image_r = get_current_images(i, proces)

            if i==1:
                VO[0].prev_disp = np.divide(VO[0].disparity.compute(prev_images_l[-1], prev_images_r[-1]).astype(np.float32), 16)
            else:
                VO[0].prev_disp = VO[-1].cur_disp
                VO[0].kp_full, VO[0].des_full = VO[-1].kp2, VO[-1].des2

            futures = [executor.submit(declare_params, j, current_image_l[j], current_image_r[j]) for j in range(proces)]
            [fut.result() for fut in futures]         

            for j in range(1, proces):
                VO[j].prev_disp = VO[j-1].cur_disp
                VO[j].kp_full, VO[j].des_full = VO[j-1].kp2, VO[j-1].des2
            
            futures = [executor.submit(VO[j].get_pose, current_image_l[j-1], current_image_l[j]) for j in range(1, proces)]
            futures.append(executor.submit(VO[0].get_pose, prev_images_l[-1], current_image_l[0]))
            transfs = [futures[j].result() for j in range(proces)]

            for tr in transfs:
                cur_pose = cur_pose @ tr
                if np.isnan(cur_pose[0, 3]):
                    cur_pose = prev_pose
                yield [cur_pose[0, 3], cur_pose[2, 3]]

            prev_images_l = current_image_l

            i += proces

if __name__ == "__main__":
    estimated_path = [pose for pose in tqdm(generate_track("08"), unit = "poses")]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=[d[0] for d in estimated_path], y=[d[1] for d in estimated_path], line=dict(width=6, color="blue"), name="estimated path"))
    fig1.update_layout(title="Visual Odometry with ORB", scene=dict(xaxis_title='x', yaxis_title='y'), width=1000, height=1000)
    fig1.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    fig1.show()

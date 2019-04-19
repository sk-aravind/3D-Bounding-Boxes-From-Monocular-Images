import cv2
import numpy as np
import os
from enum import Enum
import itertools
from .File import *
from .Utils import *

line_width = 1

class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)

def check_and_make_dir(file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise    

def constraint_to_color(constraint_idx):
    return {
        0 : cv_colors.PURPLE.value, #left
        1 : cv_colors.ORANGE.value, #top
        2 : cv_colors.MINT.value, #right
        3 : cv_colors.YELLOW.value #bottom
    }[constraint_idx]


# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])

    return pt1, pt2, pt3, pt4


# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, cam_to_img, calib_file=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)
        R0_rect = get_R0(calib_file)
        Tr_velo_to_cam = get_tr_to_velo(calib_file)

    point = np.array(pt)
    point = np.append(point, 1)

    point = np.dot(cam_to_img, point)
    # point = np.dot(np.dot(np.dot(cam_to_img, R0_rect), Tr_velo_to_cam), point)

    point = point[:2]/point[2]
    point = point.astype(np.int16)

    return point



# take in 3d points and plot them on image as red circles
def plot_3d_pts(img, pts, center, calib_file=None, cam_to_img=None, relative=False, constraint_idx=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)

    for pt in pts:
        if relative:
            pt = [i + center[j] for j,i in enumerate(pt)] # more pythonic

        point = project_3d_pt(pt, cam_to_img)

        color = cv_colors.PURPLE.value

        if constraint_idx is not None:
            color = constraint_to_color(constraint_idx)

        cv2.circle(img, (point[0], point[1]), 3, color, thickness=-1)



def plot_3d_box(img, cam_to_img, ry, dimension, center):



    R = rotation_matrix(ry)
    corners = create_corners(dimension, location=center, R=R)

    # to see the corners on image as purple circles
    # plot_3d_pts(img, corners, center,cam_to_img=cam_to_img, relative=False)

    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)
        box_3d.append(point)

    #TODO put into loop
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0],box_3d[2][1]), cv_colors.MINT.value, line_width)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.MINT.value, line_width)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), cv_colors.MINT.value, line_width)
    cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.MINT.value, line_width)

    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.MINT.value, line_width)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.MINT.value, line_width)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.MINT.value, line_width)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.MINT.value, line_width)

    for i in range(0,7,2):
        #Horizonal lines
        cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i+1][0],box_3d[i+1][1]), cv_colors.MINT.value, line_width)
    # cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[1][0],box_3d[1][1]), cv_colors.PURPLE.value, line_width)
    # cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.PURPLE.value, line_width)
    # cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.MINT.value, line_width)
    # cv2.line(img, (box_3d[6][0], box_3d[6][1]), (box_3d[7][0],box_3d[7][1]), cv_colors.MINT.value, line_width)

    front_mark = [(box_3d[i][0], box_3d[i][1]) for i in range(4)]

    cv2.line(img, front_mark[0], front_mark[3], cv_colors.PURPLE.value, 1)
    cv2.line(img, front_mark[1], front_mark[2], cv_colors.PURPLE.value, 1)

def plot_2d_box(img, box_2d):
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, 2)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, 2)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, 2)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, 2)

def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def plot_regressed_3d_bbox_2(img, truth_img, cam_to_img, box_2d, dimensions, alpha, theta_ray):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    plot_2d_box(truth_img, box_2d)
    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def rotation_matrix(yaw, pitch=0, roll=0):
    tx = roll
    ty = yaw
    tz = pitch
    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])

    return Ry.reshape([3,3])

# calib is a 3x4 matrix, box_2d is [(xmin, ymin), (xmax, ymax)]
def calc_location(dimension, proj_matrix, box_2d, alpha, theta_ray):
    #global orientation
    orient = alpha + theta_ray
    R = rotation_matrix(orient)

    # format 2d corners
    xmin = box_2d[0][0]
    ymin = box_2d[0][1]
    xmax = box_2d[1][0]
    ymax = box_2d[1][1]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]


    constraints = []
    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

   
    # negative for back and positive for front of the car
    left_mult = 1
    right_mult = -1

    # about straight on but opposite way
    if alpha < np.deg2rad(92) and alpha > np.deg2rad(88):
        left_mult = 1
        right_mult = 1
    # about straight on and same way
    elif alpha < np.deg2rad(-88) and alpha > np.deg2rad(-92):
        left_mult = -1
        right_mult = -1
    # this works but doesnt make much sense
    elif alpha < np.deg2rad(90) and alpha > -np.deg2rad(90):
        left_mult = -1
        right_mult = 1

    # if the car is facing the oppositeway, switch left and right
    switch_mult = -1
    if alpha > 0:
        switch_mult = 1

    # left and right could either be the front of the car ot the back of the car
    for i in (-1,1):
        left_constraints.append([left_mult * dx, i*dy, -switch_mult * dz])
    for i in (-1,1):
        right_constraints.append([right_mult * dx, i*dy, switch_mult * dz])

    # top and bottom are easy, just the top and bottom of car
    for i in (-1,1):
        for j in (-1,1):
            top_constraints.append([i*dx, -dy, j*dz])
    for i in (-1,1):
        for j in (-1,1):
            bottom_constraints.append([i*dx, dy, j*dz])

    # the reduced 64 options
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out repeats
    constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4,4])
    # 1's down diagonal
    for i in range(0,4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]

        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md]

        # create A, b
        A = np.zeros([4,3], dtype=np.float)
        b = np.zeros([4,1])

        indicies = [0,1,0,1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]

            # create M for corner Xx
            RX = np.dot(R, X)
            M[:3,3] = RX.reshape(3)

            M = np.dot(proj_matrix, M)

            A[row, :] = M[index,:3] - box_corners[row] * M[2,:3]
            b[row] = box_corners[row] * M[2,3] - M[index,3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        if error < best_error:
            count += 1 # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    # return best_loc, [left_constraints, right_constraints] # for debugging
    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc, best_X

def create_corners(dimension, location=None, R=None):
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

    x_corners = []
    y_corners = []
    z_corners = []

    for i in [1, -1]:
        for j in [1,-1]:
            for k in [1,-1]:
                x_corners.append(dx*i)
                y_corners.append(dy*j)
                z_corners.append(dz*k)

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in
    if R is not None:
        corners = np.dot(R, corners)

    # shift if location is passed in
    if location is not None:
        for i,loc in enumerate(location):
            corners[i,:] = corners[i,:] + loc

    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])


    return final_corners


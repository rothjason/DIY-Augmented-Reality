import cv2
import numpy as np
import matplotlib.pyplot as pyplot
import skvideo.io as skv

# Grabs user input of 20 img points, then gets input on their correspondences,
# in world coords, and saved to 'pts.txt' in input
def get_key_pts(first_im, out_file_name):
    # Select 20 points
    pyplot.imshow(first_im)
    img_pts = pyplot.ginput(n=20, timeout=120)
    pyplot.close()
    img_pts = img_pts[1:]
    
    pyplot.imshow(first_im)
    pyplot.plot([pt[0] for pt in img_pts], [pt[1] for pt in img_pts], 'bx')
    for i in range(len(img_pts)):
        pt = img_pts[i]
        pyplot.text(pt[0], pt[1], ' ' + str(i))
    pyplot.show()

    world_pts = []
    print('Please define correspondeces to world coordinates for each point! (Refer to previous img)')
    print('Enter in format "x y z". Example: "0 1 0"')
    for i in range(len(img_pts)):
        img_pt = img_pts[i]
        world_pt = str(input('Point ' + str(i) + ': ')).split(' ')
        if len(world_pt) != 3:
            print('Invalid world point! Please restart.')
            quit()
        world_pts.append(world_pt)

    f = open("input/" + out_file_name + ".txt", "w")
    for i in range(len(img_pts)):
        img_pt = img_pts[i]
        world_pt = world_pts[i]
        f.write(str(world_pt[0]) + " " + str(world_pt[1]) + " " + str(world_pt[2]) + " " + str(img_pt[0]) + " " + str(img_pt[1]) + '\n')

    f.close()

# Loads pts.txt from input and returns world points and image points (both homogenous)
# Returns: world_pts list, img_pts list
def load_key_pts(in_path):
    # Open file
    f = open(in_path, "r")
    lines = f.read().split('\n')
    lines.remove('') # delete empty rows, if any
    world_pts = []
    img_pts = []
    for line in lines:
        p = line.split(' ')
        # Parse world + img points and add homog 1
        world_pt = (int(p[0]), int(p[1]), int(p[2]), 1)
        img_pt = (int(float(p[3])), int(float(p[4])), 1)
        world_pts.append(world_pt)
        img_pts.append(img_pt)
    
    return world_pts, img_pts

# Takes in img points from video video frame and propegates to rest of video frames
# Returns: F lists of img_pts, where F is the number of frames in video_input
def propogate_keypoints(img_pts, world_pts, video_input):
    trackers = []
    for i in range(len(img_pts)):
        trackers.append(cv2.legacy.TrackerMedianFlow_create())
        
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        print("Could not open video")
        quit()

    ok, frame = video.read()
    if not ok:
        print("Can't read video")
        quit()
    bboxes = []
    for i in range(len(img_pts)):
        bboxes.append(pt_to_bbox(img_pts[i], 16, 16))
    
    for i in range(len(img_pts)):
        trackers[i].init(frame, bboxes[i])

    frames = []
    img_pts_list = [img_pts]
    while True:
        ok, frame = video.read()
        if not ok:
            break
        timer = cv2.getTickCount()
        new_img_pts = []
        for i in range(len(img_pts)):
            tracker = trackers[i]
            ok, bbox = tracker.update(frame)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            if ok: 
                # Draw rectangle:
                # p1 = (int(bbox[0]), int(bbox[1]))
                # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                # cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                new_img_pt = bbox_to_pt(bbox)
                new_img_pts.append(new_img_pt)
                # print(new_img_pt)
                cv2.circle(frame, (new_img_pt[0], new_img_pt[1]), 4, (0, 0, 255), -1)
                # cv2.putText(frame, str(world_pts[i][:3]), (new_img_pt[0], new_img_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 150, 0), 1)
            else:
                new_img_pts.append((0,0,1)) # failed (check this later)
        img_pts_list.append(new_img_pts)
        frames.append(frame)

        # ---Comment below back in to view live render of trackers! (Its a little slow)
        cv2.imshow("Tracking", frame)
        cv2.waitKey(1)
    
    
    # ---Comment below back in to save a vieo of the trackers---
    # frames = np.array(frames)
    # skv.vwrite('./output/tracking_world_coords.mp4', frames)

    return img_pts_list

# Returns top left point and height/width tuple
def pt_to_bbox(pt, width, height):
    tl = (pt[0]-(width/2), pt[1]-(height/2), 1)
    return tl[0], tl[1], width, height

# Returns homogenous point from bbox
def bbox_to_pt(bbox):
    tl = (bbox[0], bbox[1])
    width, height= bbox[2], bbox[3]
    return (int(tl[0] + (width/2)), int(tl[1] + (height/2)), 1)

# Generates projection matrix (world -> img) for single frame of video input using img_pts
# and world_pts
# Returns: Projection matrix to translate any world coords into img_coords for given frames
def calibrate_camera_matrix(world_pts, img_pts):
    # Build matrix A for Ax = 0
    A = []
    b = []
    for i in range(len(world_pts)):
        w_pt = world_pts[i]
        img_pt = img_pts[i]
        X, Y, Z = w_pt[0], w_pt[1], w_pt[2]
        u, v = img_pt[0], img_pt[1]
        A.append(np.array([X,    Y,    Z,    1.0,  0.0,  0.0,  0.0,  0.0,  -u*X,  -u*Y,  -u*Z], dtype='double'))
        A.append(np.array([0.0,  0.0,  0.0,  0.0,  X,    Y,    Z,    1.0,  -v*X,  -v*Y,  -v*Z], dtype='double'))
        b.append(u)
        b.append(v)
    A = np.array(A)
    lsq = np.linalg.lstsq(np.array(A), np.array(b), rcond=1)[0]
    lsq = np.append(lsq, 1.0)
    lsq = lsq.reshape((3, 4))
    return lsq

def render_box(video_input, world_pts, img_pts_list, out_video_name):
    big_cube_pts = [[0,0,0,1], [0,2,0,1], [2,2,0,1], [2,0,0,1],
            [0,0, 2,1],[0,2,2,1],[2,2,2,1],[2,0,2,1]]
    tiny_cube_pts = [[0,0,0,1], [0,1,0,1], [1,1,0,1], [1,0,0,1],
            [0,0,1,1],[0,1,1,1],[1,1,1,1],[1,0,1,1]]

    pts = big_cube_pts # switch between big or small cube :3
            
    output_frames = []
    for i in range(video_input.shape[0]):
        frame = video_input[i]
        img_pts = img_pts_list[i]
        transform = calibrate_camera_matrix(world_pts, img_pts)
        t_pts = []
        for pt in pts:
            t_pt = np.matmul(transform, pt)
            t_pt = (t_pt[0]/t_pt[2], t_pt[1]/t_pt[2], t_pt[2]/t_pt[2]) 
            t_pt = (int(t_pt[0]), int(t_pt[1]))
            # frame = cv2.circle(frame, t_pt, 5,(255,0,0), 6)
            t_pts.append(t_pt)

        # draw cube lines
        frame = draw_line(frame, t_pts[0], t_pts[1], (0, 255, 0))
        frame = draw_line(frame, t_pts[1], t_pts[2], (0, 255, 0))
        frame = draw_line(frame, t_pts[2], t_pts[3], (0, 255, 0))
        frame = draw_line(frame, t_pts[3], t_pts[0], (0, 255, 0))

        frame = draw_line(frame, t_pts[0], t_pts[4], (255, 0, 0))
        frame = draw_line(frame, t_pts[1], t_pts[5], (255, 0, 0))
        frame = draw_line(frame, t_pts[2], t_pts[6], (255, 0, 0))
        frame = draw_line(frame, t_pts[3], t_pts[7], (255, 0, 0))

        frame = draw_line(frame, t_pts[4], t_pts[5], (0, 0, 255))
        frame = draw_line(frame, t_pts[5], t_pts[6], (0, 0, 255))
        frame = draw_line(frame, t_pts[6], t_pts[7], (0, 0, 255))
        frame = draw_line(frame, t_pts[7], t_pts[4], (0, 0, 255))

        output_frames.append(frame)

    output = np.array(output_frames)
    skv.vwrite('./output/' + out_video_name + '.mp4', output)


# Returns img with drawn line on img between p1 and p2
def draw_line(img, p1, p2, color):
    return cv2.line(img, p1, p2, color, 5)


#-------------------------------------------------------------------------------------
# MAIN FUNCTION BELOW

# Feel free to change if you have you own video,
# just be sure to rerun get_key_pts on new video!
#
# Two existing videos are ar.mp4 and ar2.mp4, and 
# they need to pair with pts.txt and pts2.txt respectivley!
input_video_path = "./input/ar.mp4"
input_pts_path = "input/pts.txt"

video_input = skv.vread(input_video_path)

# ---Comment below back in to generate new img-world point correspondences!---
# get_key_pts(video_input[0], 'pts')

# These correspond by index!
world_pts, init_img_pts = load_key_pts(input_pts_path) # Loads saved keypoint corelations saved from get_key_pts

img_pts_list = propogate_keypoints(init_img_pts, world_pts, video_input)

render_box(video_input, world_pts, img_pts_list, 'cube_1')

#-------------------------------------------------------------------------------------
# That's it!
# DIY Augmented Reality!
This project is a simple implementation of augmented reality! It allows the user to upload their own video of a box with 3D grid points, select those points to track throughout the video, and will draw a cube (or any desired shape) on top of it! Below are the input videos, the intermediary tracking of the points, and the final projection of a 3D cube into the video!

### Final result video link:

Input 1:
https://www.youtube.com/watch?v=cBnWWU_FfEU

Input 2:
https://www.youtube.com/watch?v=zpSnsCN3nwc

[A preview of the final results!]
<img width="951" alt="Screen Shot 2023-02-23 at 6 18 46 PM" src="https://user-images.githubusercontent.com/53490165/220842033-a36d6d4d-e66c-4746-8c2e-b66d3c41c421.png">

### Point Tracking w/ world coordinates denoted

Using median-flow keypoint tracking, the user selected dots are tracked for each frame in the video:
https://www.youtube.com/watch?v=g9C7yKVu-rM

Then we can label them with pre-defined corresponding world coordinates to make a correlation between their 3D-world and 2D-video coordinates:
https://www.youtube.com/watch?v=_lFcew_DSNg

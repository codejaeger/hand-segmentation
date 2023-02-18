import cv2
import numpy as np
from pathlib import Path
import plotly_utils, preprocess_utils
import plotly.graph_objects as go
import utils

from PIL import Image, ImageDraw

kp_index = {
    'wrist': 0,              # usually called the root id
    'thumb_cmc' : 1,
    'thumb_mcp' : 2,
    'thumb_ip' : 3,
    'thumb_tip' : 4,
    'index_finger_mcp' : 5,
    'index_finger_pip' : 6,
    'index_finger_dip' : 7,
    'index_finger_tip' : 8,
    'middle_finger_mcp' : 9, # usually called the centre id
    'middle_finger_pip' : 10,
    'middle_finger_dip' : 11,
    'middle_finger_tip' : 12,
    'ring_finger_mcp' : 13,
    'ring_finger_pip' : 14,
    'ring_finger_dip' : 15,
    'ring_finger_tip' : 16,
    'pinky_finger_mcp' : 17,
    'pinky_finger_pip' : 18,
    'pinky_finger_dip' : 19,
    'pinky_finger_tip' : 20,
    'center_id': 9,
    'root_id': 0
}

def get_oriented_hand_roi(I, kp, viz=False):
    """Returns an image containing only the region defined by a crop of the hand defined by the keypoints.
    Args:
    - I: Input RGB image. Dimension: (h, w, 3)
    - kp: uv/uvd hand joint coordinates (joint numbering should conform to `kp_index` mapping). 
          If 3D then the last dimension is considered depth information. Dimension: (21, 2) or (21, 3)
    - viz: Produce intermediate plotly visualisation for debugging.
    Returns:
    - I_out: Image with same size as input image but contains only non-zero pixels for the region of interest.
    - uv_min: 2d keypoint minimum along u and v axes.
    - uv_max: 2d keypoint maximum along u and v axes.
    - u, v, d: If depth is used, a list of coordinates in the image (uvd) frame, for the non-zero 
                pixels in region of interest, all `None` otherwise.
    """
    k_uv = kp[:, :2]
    depth = True if kp.shape[-1] == 3 else False

    # Get skeleton bounds from 2d keypoints
    uv_min = np.amin(k_uv, axis=0)
    uv_max = np.amax(k_uv, axis=0)

    # Orient the image such that the hand direction (palm center to root id) is aligned to x axis.
    # This is required to get a tight crop of hand.
    orient_by = k_uv[kp_index['center_id']] - k_uv[kp_index['root_id']]
    orient_angle = np.arctan(orient_by[1] / orient_by[0])
    c, s  = np.cos(-orient_angle), np.sin(-orient_angle)
    R = np.array([[c, -s], [s, c]])
    
    # Pad image appropriately before rotating to not risk loosing out on border pixels.
    max_pad = int(max(np.linalg.norm(uv_max[0]-uv_min[0]), np.linalg.norm(uv_max[1]-uv_min[1])))
    I_padded = np.zeros(np.array(I.shape) + 2*max_pad)
    I_padded[max_pad:-max_pad, max_pad:-max_pad] = I
    I_rotated = Image.fromarray(I_padded).rotate(np.degrees(orient_angle))

    if viz:
        point_scat = plotly_utils.scatter2d(np.array([k_uv[kp_index['center_id']]+max_pad, k_uv[kp_index['root_id']]+max_pad]))
        fig1 = go.Figure([go.Heatmap(z=I_padded), point_scat])
        plotly_utils.invert_fig_y(fig1)
        plotly_utils.fit_fig_to_shape(fig1, np.array(I_rotated).shape)
        fig1.show()
    
    # Shift and rotate 2d keypoint coordinates to align with oriented image.
    k_uv = k_uv + max_pad
    k_uv_mean = np.array(I_rotated.size) / 2
    k_uv_oriented = (k_uv - k_uv_mean) @ R.T + k_uv_mean
    
    if viz:
        point_scat = plotly_utils.scatter2d(np.array(k_uv_oriented), text=21)
        fig1 = go.Figure([go.Heatmap(z=I_rotated), point_scat])
        plotly_utils.invert_fig_y(fig1)
        plotly_utils.fit_fig_to_shape(fig1, np.array(I_rotated).shape)
        fig1.show()
    
    # Find new bounding box based from oriented image.
    uv_min_oriented = np.amin(k_uv_oriented, axis=0)
    uv_max_oriented = np.amax(k_uv_oriented, axis=0)
    
    # Pad skeleton to cover hand.
    pad = 36  # pixels on each side
    u_min = int(max(uv_min_oriented[0] - pad, 0))
    v_min = int(max(uv_min_oriented[1] - pad, 0))
    u_max = int(min(uv_max_oriented[0] + pad, np.array(I_rotated).shape[1]))
    v_max = int(min(uv_max_oriented[1] + pad, np.array(I_rotated).shape[0]))

    # Crop region of interest
    I_crop = np.zeros_like(I_rotated)
    I_crop[v_min:v_max, u_min:u_max] = np.array(I_rotated)[v_min:v_max, u_min:u_max]

    I_clip = np.copy(I_crop)
    
    if depth:
        # If depth is used, apart from using uv boundaries, also make use of depth value range to clip hand.
        z_min = np.amin(kp[:, 2])
        z_max = np.amax(kp[:, 2])

        # pad skeleton to cover hand
        pad_z = 0.02    # m on each side
        z_min = z_min - (pad_z + 0.03)  # additional min_z handles offset from keypoints to hand points 
        z_max = z_max + pad_z

        I_clip[I_clip < z_min] = 0
        I_clip[I_clip > z_max] = 0
    
    # Rotate back the image to correct form.
    I_unrotated = np.array(Image.fromarray(I_clip).rotate(np.degrees(-orient_angle)))
    
    if depth:
        # If depth is used, along with the image the uvd corrdinates of non-zero pixels are also returned 
        # which must also be correctly transformed to align with original image.
        v, u = np.nonzero(I_unrotated)
        d = I_unrotated[(v, u)]
        u -= max_pad
        v -= max_pad
    else:
        u, v, d = None, None, None

    # Remove extra padding
    I_mask = I_unrotated[max_pad:-max_pad, max_pad:-max_pad]
    
    # Set all non-roi pixels to zero. 
    I_out = np.zeros_like(I)
    I_out[I_mask > 0] = I_mask[I_mask > 0]

    if viz:
        fig1 = go.Figure([go.Heatmap(z=I_clip)])
        plotly_utils.invert_fig_y(fig1)
        plotly_utils.fit_fig_to_shape(fig1, np.array(I_clip).shape)
        fig1.show()
    
    return I_out, uv_min, uv_max, (u, v, d)

def opencv_grabcut1(img, k_uv, viz=False):
    """Use binary definite foreground background separation."""
    # Start with a mask setting all pixels as probable foreground.
    mask = np.ones(img.shape[:-1]) * cv2.GC_PR_FGD
    
    # Based on region of interest (defined by k_uv), set all other pixels as definite background (GC_BGD == 0).
    mask, _, _, _ = get_oriented_hand_roi(mask, k_uv, viz)
    
    uv_min = np.amin(np.argwhere(mask == cv2.GC_PR_FGD), axis=0)
    uv_max = np.amax(np.argwhere(mask == cv2.GC_PR_FGD), axis=0)
    
    _mask = np.zeros(img.shape[:-1])
    _mask[uv_min[0]:uv_max[0], uv_min[1]:uv_max[1]] = 1
    
    mask = Image.fromarray(mask.astype(np.uint8))
    
    # Visualise
    if viz:
        fig1 = go.Figure([go.Heatmap(z=_mask)])
        plotly_utils.invert_fig_y(fig1)
        plotly_utils.fit_fig_to_shape(fig1, np.array(mask).shape)
        fig1.show()
    
    # Ignore this. Some opencv requirements.
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)
    
    img_a = np.zeros(img.shape[:-1],np.uint8)
    
    rectangle = (*uv_min[::-1], *(uv_max[::-1]-uv_min[::-1]))
    # print(rectangle)
    mask1, _, _ = cv2.grabCut(img, img_a, rectangle, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_RECT)
    
    # return mask, img_a
    return mask, mask1

def opencv_grabcut2(img, k_uv, viz=True):
    """Use probable fg-bg separation."""
    # Start with a mask setting all pixels as probable background.
    mask = np.ones(img.shape[:-1]) * cv2.GC_PR_BGD
    
    # Based on region of interest (defined by k_uv), set all other pixels as definite background (GC_BGD == 0).
    mask, _, _, _ = get_oriented_hand_roi(mask, k_uv, viz)
    mask = Image.fromarray(mask.astype(np.uint8))
    
    # Find the convex hull around the hand region defined by 2d kps.
    uvhl = cv2.convexHull(k_uv.astype(np.float32)).reshape(-1, 2)

    # Scale up this convex hull in each directions to accommodate hand shape.
    uvhl_m = np.mean(uvhl, axis=0)
    uvhl -= uvhl_m
    uvhl *= 1.4
    uvhl += uvhl_m

    _mask = Image.fromarray(np.zeros(np.array(mask).shape))
    img1 = ImageDraw.Draw(_mask)
    img1.polygon(uvhl, fill = 1, outline = 1)
    # Define skeleton for probable foreground.
    __mask = Image.fromarray(np.zeros(np.array(mask).shape))
    draw = ImageDraw.Draw(__mask)
    fl = 1
    width = int(np.linalg.norm(k_uv[1, :] - k_uv[2, :]) * 0.6)
    for i in range(21):
        draw.ellipse([(k_uv[i, 0]-3, k_uv[i, 1]-3), (k_uv[i, 0]+3, k_uv[i, 1]+3)], fill=fl)
        if i == 0:
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[1, 0], k_uv[1, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[5, 0], k_uv[5, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[9, 0], k_uv[9, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[13, 0], k_uv[13, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[17, 0], k_uv[17, 1])], fill=fl, width=width)
        elif (i) % 4 != 0:
            draw.line([(k_uv[i, 0], k_uv[i, 1]), (k_uv[i+1, 0], k_uv[i+1, 1])], fill=fl, width=width)

    # Draw a polygon within the palm region as probable foreground.
    palm = [list(k_uv[0])] + [k_uv[i*4+1] for i in range(5)]
    palm = [tuple(x) for x in palm]
    draw.polygon(palm, fill = 1, outline = 1)
    
    mask_ = np.array(__mask) * np.array(_mask)
    mask = mask * (mask_ == 0) + mask_ * cv2.GC_PR_FGD
    mask = Image.fromarray(mask.astype(np.uint8))
    
    # On the same mask now draw definite foreground regions.
    # Draw single pixel lines joining 2d joints to show hand skeleton.
    draw = ImageDraw.Draw(mask)
    fl = cv2.GC_FGD
    width = 4
    for i in range(21):
        draw.ellipse([(k_uv[i, 0]-3, k_uv[i, 1]-3), (k_uv[i, 0]+3, k_uv[i, 1]+3)], fill=fl)
        if i == 0:
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[1, 0], k_uv[1, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[5, 0], k_uv[5, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[9, 0], k_uv[9, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[13, 0], k_uv[13, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[17, 0], k_uv[17, 1])], fill=fl, width=width)
        elif (i) % 4 != 0:
            draw.line([(k_uv[i, 0], k_uv[i, 1]), (k_uv[i+1, 0], k_uv[i+1, 1])], fill=fl, width=width)

    # Draw a polygon within the palm region as definite foreground.
    palm = [list(k_uv[0])] + [k_uv[i*4+1] for i in range(5)]
    palm = [tuple(x) for x in palm]
    draw.polygon(palm, fill = cv2.GC_FGD, outline = cv2.GC_FGD)
    
    # Visualise
    if viz:
        fig1 = go.Figure([go.Heatmap(z=mask)])
        plotly_utils.invert_fig_y(fig1)
        plotly_utils.fit_fig_to_shape(fig1, np.array(mask).shape)
        fig1.show()
    
    # Ignore this. Some opencv requirements.
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)
    
    img_a = np.array(mask).astype(np.uint8)
    
    # Run for 6 iterations with initial mask.
    mask1, _, _ = cv2.grabCut(img, img_a, None, 
        backgroundModel, foregroundModel,
        6, cv2.GC_INIT_WITH_MASK)
    
    return mask, mask1

def opencv_grabcut3(img, depth_mask, k_uv, viz=False):
    """Use probable fg-bg separation and initialise with the depth mask obtained from depth data."""
    # Start with a mask setting all pixels as probable background.
    mask = np.ones(img.shape[:-1]) * cv2.GC_PR_BGD
    
    # Based on region of interest (defined by k_uv), set all other pixels as definite background (GC_BGD == 0).
    mask, _, _, _ = get_oriented_hand_roi(mask, k_uv, viz)
    mask = Image.fromarray(mask.astype(np.uint8))

    
    # Find the convex hull around the hand region defined by 2d kps.
    uvhl = cv2.convexHull(k_uv.astype(np.float32)).reshape(-1, 2)

    # Scale up this convex hull in each directions to accommodate hand shape.
    uvhl_m = np.mean(uvhl, axis=0)
    uvhl -= uvhl_m
    uvhl *= 1.4
    uvhl += uvhl_m
    
    # Draw a polygon to fill in the region defined by the convex hull of hand as probable foreground.
    _mask = Image.fromarray(np.zeros(np.array(mask).shape))
    img1 = ImageDraw.Draw(_mask)
    img1.polygon(uvhl, fill = 1, outline = 1)
    
    # Define skeleton for probable foreground.
    __mask = Image.fromarray(np.zeros(np.array(mask).shape))
    draw = ImageDraw.Draw(__mask)
    fl = 1
    width = int(np.linalg.norm(k_uv[1, :] - k_uv[2, :]) * 0.6)
    for i in range(21):
        draw.ellipse([(k_uv[i, 0]-3, k_uv[i, 1]-3), (k_uv[i, 0]+3, k_uv[i, 1]+3)], fill=fl)
        if i == 0:
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[1, 0], k_uv[1, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[5, 0], k_uv[5, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[9, 0], k_uv[9, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[13, 0], k_uv[13, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[17, 0], k_uv[17, 1])], fill=fl, width=width)
        elif (i) % 4 != 0:
            draw.line([(k_uv[i, 0], k_uv[i, 1]), (k_uv[i+1, 0], k_uv[i+1, 1])], fill=fl, width=width)

    # Draw a polygon within the palm region as probable foreground.
    palm = [list(k_uv[0])] + [k_uv[i*4+1] for i in range(5)]
    palm = [tuple(x) for x in palm]
    draw.polygon(palm, fill = 1, outline = 1)
    
    mask_ = np.array(__mask) * np.array(_mask)
    mask = mask * (mask_ == 0) + mask_ * cv2.GC_PR_FGD

    if viz:
        fig1 = go.Figure([go.Heatmap(z=__mask)])
        plotly_utils.invert_fig_y(fig1)
        plotly_utils.fit_fig_to_shape(fig1, np.array(mask).shape)
        fig1.show()
    
    msk = np.copy(np.array(mask))
#     print(np.array(mask), depth_mask==0, cv2.GC_BGD, cv2.GC_PR_BGD)
#     print(np.sum(np.array(mask)==cv2.GC_BGD), np.sum(np.logical_and(mask==cv2.GC_BGD, depth_mask==0)))
    
    # case 1: definite bg & definite bg(0) = dfg
    msk[np.logical_and(mask==cv2.GC_BGD, depth_mask==0)] = cv2.GC_BGD
    
    # case 2: definite bg & probable fg(1) = dfg
    msk[np.logical_and(mask==cv2.GC_BGD, depth_mask==1)] = cv2.GC_BGD
    
    # case 3: probable bg & definite bg = dfg
    msk[np.logical_and(mask==cv2.GC_PR_BGD, depth_mask==0)] = cv2.GC_BGD
        
    # case 4: probable bg & probable fg = pfg
    msk[np.logical_and(mask==cv2.GC_PR_BGD, depth_mask==1)] = cv2.GC_PR_FGD
    
    # case 5: probable fg & definite bg = pbg
    msk[np.logical_and(mask==cv2.GC_PR_FGD, depth_mask==0)] = cv2.GC_PR_BGD
    
    # case 6: probable fg & probable fg = pfg
    msk[np.logical_and(mask==cv2.GC_PR_FGD, depth_mask==1)] = cv2.GC_PR_FGD
    
    mask = Image.fromarray(msk.astype(np.uint8))
    
    # On the same mask now draw definite foreground regions.
    # Draw single pixel lines joining 2d joints to show hand skeleton.
    draw = ImageDraw.Draw(mask)
    fl = cv2.GC_FGD
    width = 4
    for i in range(21):
        draw.ellipse([(k_uv[i, 0]-3, k_uv[i, 1]-3), (k_uv[i, 0]+3, k_uv[i, 1]+3)], fill=fl)
        if i == 0:
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[1, 0], k_uv[1, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[5, 0], k_uv[5, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[9, 0], k_uv[9, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[13, 0], k_uv[13, 1])], fill=fl, width=width)
            draw.line([(k_uv[0, 0], k_uv[0, 1]), (k_uv[17, 0], k_uv[17, 1])], fill=fl, width=width)
        elif (i) % 4 != 0:
            draw.line([(k_uv[i, 0], k_uv[i, 1]), (k_uv[i+1, 0], k_uv[i+1, 1])], fill=fl, width=width)

    # Draw a polygon within the palm region as definite foreground.
    palm = [list(k_uv[0])] + [k_uv[i*4+1] for i in range(5)]
    palm = [tuple(x) for x in palm]
    draw.polygon(palm, fill = cv2.GC_FGD, outline = cv2.GC_FGD)
    
    # Visualise
    if viz:
        fig1 = go.Figure([go.Heatmap(z=mask)])
        plotly_utils.invert_fig_y(fig1)
        plotly_utils.fit_fig_to_shape(fig1, np.array(mask).shape)
        fig1.show()
    
    # Ignore this. Some opencv requirements.
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)
    
    img_a = np.array(mask)
    
    # Run for 6 iterations with initial mask.
    mask1, _, _ = cv2.grabCut(img, img_a, None, 
        backgroundModel, foregroundModel,
        6, cv2.GC_INIT_WITH_MASK)
    
    return mask, mask1

# def get_mask(
#     rgb,
#     depth=None,
#     rgb_keypoints=None,
#     depth_keypoints=None,
#     rgb_intrinsic_mat=None,
#     depth_intrinsic_mat=None,
#     rgb_to_depth_mat=None, # 4x4 matrix - M @ v
#     output_frame_of_reference='rgb',
#     grabcut_version=2,
#     viz=False,
#     save_mask_at=""
# ):
#     """All arrays and matrices should be in the numpy format."""
#     # Generate RGB keypoints in uv format if not provided
#     if not rgb_keypoints:
#         # Use mediapipe
#         pass
#     elif not rgb_intrinsic_mat:
#         rgb_kp_uv = rgb_keypoints[:, :2]
#     else:
#         fx_color, fy_color, cx_color, cy_color = utils.get_intrinsic_params(rgb_intrinsic_mat)
#         rgb_kp_uv = preprocess_utils.xyz2uv(rgb_keypoints, fx_color, fy_color, cx_color, cy_color)

#     # Generate depth keypoints in uvd format if not provided
#     if not depth_keypoints:
#         # Use AWR 
#         pass
#     elif not depth_intrinsic_mat:
#         depth_kp_uvd = depth_keypoints
#     else:
#         fx_depth, fy_depth, cx_depth, cy_depth = utils.get_intrinsic_params(depth_intrinsic_mat)
#         depth_kp_uv = preprocess_utils.xyz2uv(depth_keypoints, fx_depth, fy_depth, cx_depth, cy_depth)
#         depth_kp_uvd = np.concatenate([depth_kp_uv, depth_keypoints[:, 2]], axis=-1)

    

# def get_mask(sample_id, viz=False, use_depth=False):
#     k_xyz_mm_wrt_depth = K_xyz_mm_wrt_depth[sample_id, :, :]  # (21, 3)
#     rot_mat, translation = get_intel_color_extrinsic_params()
#     k_xyz_mm_wrt_color = k_xyz_mm_wrt_depth @ rot_mat - translation  # (21, 3)

#     fx_color, fy_color, cx_color, cy_color = get_intel_color_intrinsic_params()
#     fx_depth, fy_depth, cx_depth, cy_depth = get_intel_depth_intrinsic_params()
    
#     # get uv corrdinates for color camera frame.
#     k_uv = preprocess_utils.xyz2uv(k_xyz_mm_wrt_color, fx_color, fy_color, cx_color, cy_color) 
    
#     # get uv corrdinates for depth camera frame.
#     k_uv_depth = preprocess_utils.xyz2uv(k_xyz_mm_wrt_depth, fx_depth, fy_depth, cx_depth, cy_depth) 
    
#     # load rgb
#     cv_img = cv2.imread(str(path_to_images / f'SK_color_{sample_id}.png'))
    
#     # load depth
#     depth_png = np.array(Image.open(path_to_images / f'SK_depth_{sample_id}.png'))
#     depth = (depth_png[:, :, 0] + depth_png[:, :, 1].astype(np.float32)*256).astype(np.float32) # (480, 640)

#     # mm to m
#     depth = depth / 1000
    
#     k_data = k_xyz_mm_wrt_depth / 1000

#     if viz:
#         fig1 = go.Figure([go.Image(z=np.array(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)))])
#         plotly_utils.invert_fig_y(fig1)
#         plotly_utils.fit_fig_to_shape(fig1, np.array(cv_img).shape)
#         fig1.show()

#     _, mask1 = opencv_grabcut(cv_img, k_uv, viz)

#     mask1 = np.array(mask1)
#     # Choose predicted probable and definite foreground as mask region
#     msk = np.logical_or(mask1==3, mask1==1)

#     if viz:
#         fig1 = go.Figure([go.Heatmap(z=mask1)])
#         plotly_utils.invert_fig_y(fig1)
#         plotly_utils.fit_fig_to_shape(fig1, np.array(msk).shape)
#         fig1.show()

#     if use_depth:
#         # If depth values are available, get a mask using the uv coords and depth value range.
#         mask2, _, _, uvd  = get_oriented_hand_roi(depth, k_uv_depth, k_xyz=k_data, depth=True)

#         if viz:
#             fig2 = go.Figure([go.Heatmap(z=mask2)])
#             plotly_utils.invert_fig_y(fig2)
#             plotly_utils.fit_fig_to_shape(fig2, np.array(mask2).shape)
#             fig2.show()

#         u, v, d = uvd

#         # convert point coordinates from depth uv to color uv
#         # uvd depth to xyz depth
#         xyz_depth = preprocess_utils.uvd2xyz(np.concatenate([u[:, np.newaxis], v[:, np.newaxis], d[:, np.newaxis]], axis=-1), fx_depth, fy_depth, cx_depth, cy_depth)

#         # xyz depth to xyz color
#         xyz_color = (xyz_depth * 1000) @ rot_mat - translation

#         # xyz color to uv color
#         uv_color = preprocess_utils.xyz2uv(xyz_color, fx_color, fy_color, cx_color, cy_color).astype(np.int32)

#         msk2 = np.zeros(msk.shape)

#         # When converting uv coordinates from depth to color camera frame, some pixels inside hand may be
#         # lost. Include all the (8) neighboring pixels of depth mask pixels into the mask.
#         shifts = np.array([(-1, -1), (-1, 1), (-1, 0), (1, -1), (1, 1), (1, 0), (0, 1), (0, -1)])
#         uv_all = np.tile(uv_color, [8, 1, 1])
#         uv_all += shifts[:, np.newaxis, :]
#         uv_all = uv_all.reshape([-1, 2])

#         msk2[(uv_all[:, 1], uv_all[:, 0])] = 1
        
#         if viz:
#             fig1 = go.Figure([go.Heatmap(z=msk2)])
#             plotly_utils.invert_fig_y(fig1)
#             plotly_utils.fit_fig_to_shape(fig1, np.array(msk2).shape)
#             fig1.show()

#             fig1 = go.Figure([go.Heatmap(z=msk*1.0)])
#             plotly_utils.invert_fig_y(fig1)
#             plotly_utils.fit_fig_to_shape(fig1, np.array(msk2).shape)
#             fig1.show()

#         # combine two masks - gets rid of wrist and extra regions considered by grabcut
#         msk = msk * msk2

#     masked = Image.fromarray((msk.reshape(480, 640, 1) * np.array(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))).astype(np.uint8))
#     if viz:
#         display(masked)

#     return msk

def get_mediapipe_kps(image_path, view="first"):
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
    static_image_mode=True,      # False to treat inputs as sequence of frames
    max_num_hands=4,             # Max number of hands tracked (w/wo) using tracking
    model_complexity=1,          # Higher model complexity means higher inference times
    min_detection_confidence=0.5 # 
    # min_tracking_confidence=0.5 # Tracking accuracy might increase at the expense of latency  
    ) as hands:
        image = cv2.imread(image_path)
        if view == "third":
            image = cv2.flip(image, 1)
        # Convert the BGR image to RGB before processing.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

    if not results.multi_hand_landmarks:
        return []
    image_height, image_width, _ = image.shape
    print("Multi-handedness", results.multi_handedness)
    kps = []
    for hand_landmarks in results.multi_hand_landmarks:
        kp1 = []
        for ind in range(21):
            kp1.append((hand_landmarks.landmark[ind].x * image_width, hand_landmarks.landmark[ind].y * image_height, hand_landmarks.landmark[ind].z))
        kps.append(kp1)
    return kps

if __name__ == '__main__':

    import glob
    IMAGE_FILES = sorted(glob.glob('../../../test_images/*.jpg'))
    
    for img_path in IMAGE_FILES:
        print(img_path)
        kps = get_mediapipe_kps(img_path)

        # print(np.array(kp1).shape, np.array(kp2).shape)
        orig_img = Image.open(img_path)
        msk = np.zeros(np.array(orig_img).shape[:2])

        for kp in kps:
            _, mask1 = opencv_grabcut2(np.array(orig_img), np.array(kp)[:, :2], False)
            msk1 = np.logical_or(mask1==3, mask1==1)

            msk = np.logical_or(msk, msk1)
        rgb_format = np.array(orig_img) * np.expand_dims(msk, axis=-1)
        # rgb_format = np.ones_like(orig_img) * np.expand_dims(msk, axis=-1) * 255

        Image.fromarray(rgb_format.astype(np.uint8)).save(f"/Users/mac/predict3/{img_path.split('/')[-1].split('.')[0]}.png")
        # break


    # sample_id = [253, 401, 430, 445, 489, 504, 729]
    # # sample_id = 1021
    # for i in sample_id:
    #     path_to_stb = Path('../../../datasets/stb')
    #     path_to_images = path_to_stb / 'images' / 'B1Counting'
    #     path_to_labels = path_to_stb / 'labels'

    #     mat_file = sio.loadmat(path_to_labels / 'B1Counting_SK.mat')
    #     K_xyz_mm_wrt_depth = mat_file['handPara']    # (3, 21, 1500)
    #     K_xyz_mm_wrt_depth = np.transpose(K_xyz_mm_wrt_depth, [2, 1, 0])  # (1500, 21, 3)

    #     mask = get_mask(i, viz=False, use_depth=True)
    #     rgb_format = np.ones((*mask.shape, 3)) * (255, 255, 255) * np.expand_dims(mask, axis=-1)

    #     Image.fromarray(rgb_format.astype(np.uint8)).save(f'/Users/mac/predict/{i}.png')


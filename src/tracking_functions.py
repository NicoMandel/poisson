# tracking_functions.py
import cv2
import numpy as np

def _subpixel_refinement(corr_map, max_loc):
    """
    Refines the peak location in a correlation map to sub-pixel accuracy.
    This uses a 2D quadratic fit around the peak.
    """
    cx, cy = max_loc
    if cx > 0 and cx < corr_map.shape[1] - 1 and cy > 0 and cy < corr_map.shape[0] - 1:
        y, x = np.mgrid[-1:2, -1:2]
        sub_map = corr_map[cy-1:cy+2, cx-1:cx+2]
        
        # Fit a 2D quadratic: z = a*x^2 + b*y^2 + c*xy + d*x + e*y + f
        A = np.vstack([x.ravel()**2, y.ravel()**2, x.ravel()*y.ravel(), x.ravel(), y.ravel(), np.ones(9)]).T
        b = sub_map.ravel()
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            a, b, c, d, e, _ = coeffs
            
            # Find the peak of the quadratic surface
            x_offset = (2*b*d - c*e) / (c**2 - 4*a*b)
            y_offset = (2*a*e - c*d) / (c**2 - 4*a*b)

            # Ensure the offset is within a reasonable range
            if abs(x_offset) < 1 and abs(y_offset) < 1:
                return cx + x_offset, cy + y_offset
        except np.linalg.LinAlgError:
            pass # Fallback to integer location if fit fails
            
    return float(cx), float(cy)

def track_subset_ncc_roi(ref_gray, cur_gray, original_pos, last_pos, subset_size, search_margin=30, use_blur=False):
    """
    Tracks using NCC but limits the search to a Region of Interest (ROI) around the last known position.
    """
    if use_blur:
        ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)
        cur_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)

    half_size = subset_size // 2
    
    # 1. Grab the template from the REFERENCE frame
    tx, ty = int(original_pos[0]), int(original_pos[1])
    
    # Ensure template is within image bounds
    if ty - half_size < 0 or tx - half_size < 0 or ty + half_size >= ref_gray.shape[0] or tx + half_size >= ref_gray.shape[1]:
        return last_pos, 0.0
        
    template = ref_gray[ty - half_size : ty + half_size + 1, tx - half_size : tx + half_size + 1]

    # 2. Define the search ROI in the CURRENT frame
    lx, ly = int(last_pos[0]), int(last_pos[1])
    y_min = max(0, ly - half_size - search_margin)
    y_max = min(cur_gray.shape[0], ly + half_size + search_margin + 1)
    x_min = max(0, lx - half_size - search_margin)
    x_max = min(cur_gray.shape[1], lx + half_size + search_margin + 1)
    
    search_roi = cur_gray[y_min:y_max, x_min:x_max]

    # Handle edge case where search window is smaller than template
    if search_roi.shape[0] < template.shape[0] or search_roi.shape[1] < template.shape[1]:
        return last_pos, 0.0

    # 3. Match template within the smaller ROI
    res = cv2.matchTemplate(search_roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    # 4. Sub-pixel refinement
    refined_x, refined_y = _subpixel_refinement(res, max_loc)

    # 5. Map the local ROI coordinates back to global image coordinates
    global_center_x = (x_min + refined_x) + half_size
    global_center_y = (y_min + refined_y) + half_size

    return (global_center_x, global_center_y), max_val

def track_subset_lk(prev_gray, cur_gray, prev_points, use_blur=False):
    """
    Tracks points using Lucas-Kanade Optical Flow strictly frame-to-frame.
    """
    if use_blur:
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        cur_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_points, None, **lk_params)
    return new_points, status.ravel()

def track_subset_ecc(ref_gray, cur_gray, original_pos, subset_size, use_blur=False):
    """
    Tracks using Enhanced Correlation Coefficient (ECC) Maximization handling affine deformations.
    """
    if use_blur:
        ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)
        cur_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)

    half_size = subset_size // 2
    tx, ty = int(original_pos[0]), int(original_pos[1])
    
    # Boundary checks
    if ty-half_size < 0 or tx-half_size < 0 or ty+half_size >= ref_gray.shape[0] or tx+half_size >= ref_gray.shape[1]:
        return original_pos, 0.0

    template = ref_gray[ty-half_size : ty+half_size, tx-half_size : tx+half_size]
    
    # ROI in current image (make it slightly larger than template to catch motion)
    margin = 15
    y_min, y_max = max(0, ty-half_size-margin), min(cur_gray.shape[0], ty+half_size+margin)
    x_min, x_max = max(0, tx-half_size-margin), min(cur_gray.shape[1], tx+half_size+margin)
    roi = cur_gray[y_min:y_max, x_min:x_max]
    
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
        return original_pos, 0.0

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    
    try:
        score, warp_matrix = cv2.findTransformECC(template, roi, warp_matrix, cv2.MOTION_AFFINE, criteria, None, 1)
        # Apply transformation to the center of the template
        new_x = x_min + half_size + warp_matrix[0, 2] 
        new_y = y_min + half_size + warp_matrix[1, 2]
        return (new_x, new_y), score
    except Exception as e:
        return original_pos, 0.0 # Fallback if ECC fails to converge
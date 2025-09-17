# tracking_functions.py
import cv2
import numpy as np

def _subpixel_refinement(corr_map, max_loc):
    """
    Refines the peak location in a correlation map to sub-pixel accuracy.
    This uses a 2D quadratic fit around the peak.
    """
    # Get the 3x3 region around the peak
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
            
    return float(cx), float(cy) # Return integer location if at edge or fit fails


def track_subset_ncc(ref_gray, cur_gray, last_pos, subset_size, use_blur=False):
    """
    Tracks a subset using Normalized Cross-Correlation with sub-pixel refinement.
    """
    if use_blur:
        ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)
        cur_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)

    half_size = subset_size // 2
    x_start, y_start = int(last_pos[0] - half_size), int(last_pos[1] - half_size)
    
    if y_start < 0 or x_start < 0 or y_start + subset_size >= ref_gray.shape[0] or x_start + subset_size >= ref_gray.shape[1]:
        return last_pos, 0.0
        
    template = ref_gray[y_start:y_start + subset_size, x_start:x_start + subset_size]
    
    res = cv2.matchTemplate(cur_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Get refined sub-pixel location
    refined_x, refined_y = _subpixel_refinement(res, max_loc)
    
    # Convert from top-left corner to center position
    new_center_x = refined_x + half_size
    new_center_y = refined_y + half_size
    
    return (new_center_x, new_center_y), max_val


def track_subset_lk(ref_gray, cur_gray, last_known_points, use_blur=False):
    """
    Tracks points using Lucas-Kanade Optical Flow.
    'last_known_points' should be a NumPy array of shape (n, 1, 2).
    """
    if use_blur:
        ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)
        cur_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Calculate optical flow
    new_points, status, error = cv2.calcOpticalFlowPyrLK(ref_gray, cur_gray, last_known_points, None, **lk_params)
    
    # Return the new points and the status (1 for successfully tracked points)
    return new_points, status.ravel()
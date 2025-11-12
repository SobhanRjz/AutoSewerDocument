"""Robust pipe rim detection with temporal filtering."""

import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Protocol

# OpenCV optimizations
cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, os.cpu_count() - 1))


@dataclass
class PipeRim:
    """Detected pipe rim geometry."""
    center_x: float
    center_y: float
    radius: float
    confidence: float = 1.0
    is_ellipse: bool = False
    major_axis: Optional[float] = None
    minor_axis: Optional[float] = None
    angle: Optional[float] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)


class IRimDetector(Protocol):
    """Interface for pipe rim detection."""
    def detect(self, frame: np.ndarray) -> Optional[PipeRim]: ...


class PipeRimDetector:
    """Detect pipe rim using advanced gradient-based refinement and ellipse fitting."""
    
    def __init__(
        self,
        edge_low: int = 50,
        edge_high: int = 150,
        min_radius: int = 100,
        max_radius: int = 1000,
        ransac_iterations: int = 200,
        ransac_threshold: float = 5.0,
        temporal_alpha: float = 0.7,
        use_hough: bool = True,
        blur_kernel: int = 5,
        use_clahe: bool = True,
        use_gradient_refinement: bool = True,
        use_polar_refinement: bool = True,
        use_ellipse_fit: bool = True,
        use_search_band: bool = True,
        use_frst_center: bool = True,
        pyramid_scale: float = 0.5,
        confidence_threshold: float = 0.9,
        max_gradient_points: int = 5000
    ):
        """
        Args:
            edge_low: Canny lower threshold (ignored if auto_canny enabled)
            edge_high: Canny upper threshold (ignored if auto_canny enabled)
            min_radius: Minimum pipe radius in pixels
            max_radius: Maximum pipe radius in pixels
            ransac_iterations: RANSAC iteration count (default 200 for speed)
            ransac_threshold: RANSAC inlier distance threshold
            temporal_alpha: EMA smoothing [0,1]. Higher = more smoothing
            use_hough: Use Hough Circle Transform for initial estimate
            blur_kernel: Gaussian blur kernel size for preprocessing
            use_clahe: Apply CLAHE contrast enhancement
            use_gradient_refinement: Refine center using gradient voting
            use_polar_refinement: Refine radius using polar transform
            use_ellipse_fit: Attempt ellipse fitting for off-axis cameras
            use_search_band: Use temporal search band to reduce noise
            use_frst_center: Use Fast Radial Symmetry Transform for center detection
            pyramid_scale: Scale for image pyramid (0.5 = 4x speedup)
            confidence_threshold: Skip heavy ops if confidence above this (default 0.9)
            max_gradient_points: Max points for gradient refinement (cap memory)
        """
        self._edge_low = edge_low
        self._edge_high = edge_high
        self._min_radius = min_radius
        self._max_radius = max_radius
        self._ransac_iterations = ransac_iterations
        self._ransac_threshold = ransac_threshold
        self._temporal_alpha = temporal_alpha
        self._use_hough = use_hough
        self._blur_kernel = blur_kernel
        self._use_clahe = use_clahe
        self._use_gradient_refinement = use_gradient_refinement
        self._use_polar_refinement = use_polar_refinement
        self._use_ellipse_fit = use_ellipse_fit
        self._use_search_band = use_search_band
        self._use_frst_center = use_frst_center
        self._pyramid_scale = pyramid_scale
        self._confidence_threshold = confidence_threshold
        self._max_gradient_points = max_gradient_points
        
        self._prev_rim: Optional[PipeRim] = None
        self._stable_frames = 0
    
    def detect(self, frame: np.ndarray) -> Optional[PipeRim]:
        """
        Detect pipe rim with pyramid optimization and fast-path for stable tracking.
        
        Args:
            frame: BGR input frame
            
        Returns:
            Detected rim or None if detection fails
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing
        gray = self._preprocess(gray)
        
        # Get edge image
        edges = self._auto_canny(gray)
        
        # Apply search band if previous rim exists
        if self._use_search_band and self._prev_rim is not None:
            edges = self._apply_search_band(edges, self._prev_rim)
        
        # Try ellipse first if camera is off-axis (first frame only)
        rim = None
        if self._use_ellipse_fit and self._prev_rim is None:
            # Check if camera appears off-axis by looking for asymmetric edge distribution
            h, w = gray.shape[:2]
            test_edges = self._auto_canny(gray)
            left_edges = test_edges[:, :w//2].sum()
            right_edges = test_edges[:, w//2:].sum()
            asymmetry = abs(left_edges - right_edges) / max(left_edges + right_edges, 1)
            if asymmetry > 0.3:  # significant asymmetry suggests off-axis
                rim = self._detect_ellipse(test_edges)

        # Fast FRST + polar method (alternative to Hough)
        if rim is None and self._use_frst_center:
            try:
                # FRST center detection
                cx, cy = self._frst_center(gray)

                # Optional small ROI around FRST center for speed on next step
                roi_size = int(min(gray.shape) * 0.3)  # 30% of smaller dimension
                x0 = max(0, int(cx - roi_size//2))
                y0 = max(0, int(cy - roi_size//2))
                x1 = min(gray.shape[1], x0 + roi_size)
                y1 = min(gray.shape[0], y0 + roi_size)
                roi_gray = gray[y0:y1, x0:x1]
                roi_cx, roi_cy = cx - x0, cy - y0

                # Polar radius detection in ROI
                r = self._radius_from_polar_robust(roi_gray, roi_cx, roi_cy, self._min_radius, self._max_radius)

                # geometric guard: don't allow circles near frame border
                max_border = 0.85 * min(cx, cy, gray.shape[1]-cx, gray.shape[0]-cy)  # was 0.90
                r = min(r, max_border)

                # temporal prior: tighter once stable
                if self._prev_rim is not None:
                    tight = 0.08 if self._stable_frames >= 5 else 0.12
                    r_min_p = max(self._min_radius, (1.0 - tight) * self._prev_rim.radius)
                    r_max_p = min(self._max_radius, (1.0 + tight) * self._prev_rim.radius)
                    r = float(np.clip(r, r_min_p, r_max_p))

                # final validation (same metrics as _pick_best_circle)
                if self._validate_circle_candidate(gray, cx, cy, r):
                    rim = PipeRim(cx, cy, r, confidence=0.9, is_ellipse=False)
                    self._stable_frames = min(self._stable_frames + 1, 10)  # cap at 10
                else:
                    self._stable_frames = 0
            except:
                pass  # Fall back to other methods

        # Ellipse fallback threshold: if off-center FRST center, try ellipse first
        if rim is None and self._use_ellipse_fit and self._use_frst_center:
            try:
                cx_frst, cy_frst = self._frst_center(gray)
                h, w = gray.shape[:2]
                cx_frame, cy_frame = w/2, h/2
                dist_from_center = np.sqrt((cx_frst - cx_frame)**2 + (cy_frst - cy_frame)**2)
                # Try ellipse if significantly off-center (more than 35% of expected radius range)
                expected_r = (self._min_radius + self._max_radius) / 2
                if dist_from_center / expected_r > 0.35:
                    ellipse_rim = self._detect_ellipse(edges)
                    if ellipse_rim is not None and self._validate_circle_candidate(gray, ellipse_rim.center_x, ellipse_rim.center_y, ellipse_rim.radius):
                        rim = ellipse_rim
            except:
                pass  # FRST failed, skip ellipse fallback

        # Try Hough or ellipse fitting
        if rim is None and self._use_hough:
            rim = self._detect_hough(gray, edges)

        # Try ellipse fitting if circle detection fails
        if rim is None and self._use_ellipse_fit:
            rim = self._detect_ellipse(edges)

        # Fallback to RANSAC
        if rim is None:
            edge_points = np.column_stack(np.where(edges > 0))
            if len(edge_points) >= 3:
                rim = self._ransac_circle_fit(edge_points)

        if rim is None:
            return self._prev_rim

        # Skip expensive refinement for high-confidence Hough results
        rim_confidence = getattr(rim, "confidence", 0.0) or 0.0
        needs_refinement = rim_confidence < self._confidence_threshold

        # Apply gradient refinement only if needed
        if self._use_gradient_refinement and not rim.is_ellipse and needs_refinement:
            cx, cy = self._refine_center_by_gradient(gray, rim.center_x, rim.center_y, rim.radius)
            rim.center_x, rim.center_y = cx, cy

        # Apply polar refinement only if needed
        if self._use_polar_refinement and not rim.is_ellipse and needs_refinement:
            r = self._polar_radius_search(gray, rim.center_x, rim.center_y, rim.radius)
            if self._min_radius <= r <= self._max_radius:
                rim.radius = r
        
        # Temporal smoothing
        if self._prev_rim is not None:
            rim = self._smooth_temporal(rim, self._prev_rim)
        
        self._prev_rim = rim
        return rim
    
    def reset(self) -> None:
        """Reset temporal state."""
        self._prev_rim = None
    
    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing with CLAHE and sharpening."""
        if self._use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Blur + sharpen
        if self._blur_kernel > 0:
            blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
            sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
            return sharp
        
        return gray
    
    def _auto_canny(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive Canny edge detection based on median intensity."""
        v = np.median(gray)
        low = int(max(0, 0.66 * v))
        high = int(min(255, 1.33 * v))
        return cv2.Canny(gray, low, high)
    
    def _apply_search_band(self, edges: np.ndarray, prev_rim: PipeRim) -> np.ndarray:
        """Mask edges to narrow band around previous rim location."""
        h, w = edges.shape
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - prev_rim.center_x)**2 + (yy - prev_rim.center_y)**2)

        # Temporal band tightening: ±10% when stable, ±15% when unstable
        band_width = 0.10 if self._stable_frames >= 5 else 0.15
        inner = prev_rim.radius * (1 - band_width)
        outer = prev_rim.radius * (1 + band_width)

        mask = (dist >= inner) & (dist <= outer)
        return np.where(mask, edges, 0).astype(np.uint8)
    
    def _refine_center_by_gradient(
        self, gray: np.ndarray, cx: float, cy: float, r: float,
        span_ratio: float = 0.12, step: int = 2
    ) -> Tuple[float, float]:
        """Fast center refinement using gradient-weighted edge point analysis."""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)
        
        # Extract ring of edge points near expected radius
        h, w = gray.shape
        yy, xx = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        
        # Create ring mask
        ring_mask = np.abs(dist_from_center - r) < (0.1 * r)
        
        # Strong gradients only
        strong_edges = mag > np.percentile(mag, 85)
        rim_points = ring_mask & strong_edges
        
        if not np.any(rim_points):
            return cx, cy
        
        # Get coordinates and gradients of rim points
        y_pts, x_pts = np.where(rim_points)
        gx_pts = gx[rim_points]
        gy_pts = gy[rim_points]
        mag_pts = mag[rim_points]
        
        # Each gradient should point radially from center
        # Calculate where each gradient ray intersects with origin rays
        dx = x_pts - cx
        dy = y_pts - cy
        
        # Gradient alignment score (should point outward from center)
        alignment = (gx_pts * dx + gy_pts * dy) / (mag_pts * np.sqrt(dx*dx + dy*dy) + 1e-6)
        
        # Keep only points with outward-pointing gradients
        valid = alignment > 0.3
        if not np.any(valid):
            return cx, cy
        
        x_valid = x_pts[valid]
        y_valid = y_pts[valid]
        gx_valid = gx_pts[valid]
        gy_valid = gy_pts[valid]
        mag_valid = mag_pts[valid]
        
        # Backproject gradients to find center
        # Center is where gradient rays converge
        cx_votes = x_valid - gx_valid * r / (mag_valid + 1e-6)
        cy_votes = y_valid - gy_valid * r / (mag_valid + 1e-6)
        
        # Weighted average
        weights = mag_valid
        new_cx = np.average(cx_votes, weights=weights)
        new_cy = np.average(cy_votes, weights=weights)
        
        # Clip to reasonable range
        max_shift = r * span_ratio
        new_cx = np.clip(new_cx, cx - max_shift, cx + max_shift)
        new_cy = np.clip(new_cy, cy - max_shift, cy + max_shift)
        
        return float(new_cx), float(new_cy)
    
    def _polar_radius_search(
        self, gray: np.ndarray, cx: float, cy: float, r_guess: float, delta: float = 0.15
    ) -> float:
        """Refine radius using polar transform."""
        max_r = int(min(cx, cy, gray.shape[1] - cx, gray.shape[0] - cy))
        R = int(min(max_r, r_guess * (1 + delta)))
        
        if R <= 0:
            return r_guess
        
        try:
            polar = cv2.warpPolar(gray, (720, R), (cx, cy), R, cv2.WARP_POLAR_LINEAR)
            g = cv2.Sobel(polar, cv2.CV_32F, 1, 0, ksize=3)
            prof = np.mean(np.abs(g), axis=0)
            r_idx = int(np.argmax(prof))
            return float(r_idx)
        except cv2.error:
            return r_guess
    
    def _detect_ellipse(self, edges: np.ndarray) -> Optional[PipeRim]:
        """Detect elliptical rim for off-axis cameras."""
        edge_points = np.column_stack(np.where(edges > 0))
        
        if len(edge_points) < 5:
            return None
        
        # Convert from (y,x) to (x,y) format for fitEllipse
        points = np.column_stack([edge_points[:, 1], edge_points[:, 0]]).astype(np.float32)
        
        try:
            ellipse = cv2.fitEllipse(points)
            (cx, cy), (ma, mi), angle = ellipse
            
            # Compute equivalent radius
            r_equiv = (ma + mi) / 4.0
            
            if not (self._min_radius <= r_equiv <= self._max_radius):
                return None
            
            return PipeRim(
                center_x=float(cx),
                center_y=float(cy),
                radius=float(r_equiv),
                confidence=1.0,
                is_ellipse=True,
                major_axis=float(ma / 2),
                minor_axis=float(mi / 2),
                angle=float(angle)
            )
        except cv2.error:
            return None
    
    def _detect_hough(self, gray: np.ndarray, edges: np.ndarray | None = None) -> Optional[PipeRim]:
        """
        Fast + robust Hough pass:
        - optional downscale (coarse)
        - tight radius range (from previous rim if available)
        - ROI instead of full frame when tracking
        - higher dp + larger minDist
        - stronger param2 to reduce candidates
        - median blur (helps Hough a lot)
        """
        g = gray
        if g.dtype != np.uint8:
            g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        h, w = g.shape[:2]
        use_roi = getattr(self, "_prev_rim", None) is not None

        # ---- 1) If we have a previous detection, search only in a small ROI & tight radius band
        if use_roi:
            pr = self._prev_rim
            cx, cy, r = int(pr.center_x), int(pr.center_y), int(pr.radius)
            pad = int(0.18 * r)                               # tighten if very stable: 0.10–0.15
            x0, y0 = max(0, cx - r - pad), max(0, cy - r - pad)
            x1, y1 = min(w, cx + r + pad), min(h, cy + r + pad)
            roi = g[y0:y1, x0:x1]

            # radius band ~ ±15% around last radius
            minR = max(self._min_radius, int(r * 0.85))
            maxR = min(self._max_radius, int(r * 1.15))

            # small downscale inside ROI (huge speedup; we'll scale back)
            scale = 0.5 if roi.size > 640*640 else 1.0
            if scale != 1.0:
                roi_s = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                minR_s, maxR_s = int(minR * scale), int(maxR * scale)
            else:
                roi_s, minR_s, maxR_s = roi, minR, maxR

            # median blur helps Hough a lot
            roi_s = cv2.medianBlur(roi_s, 5)

            # faster accumulator: dp > 1, and large minDist so we don't get tons of near-duplicate circles
            dp = 1.4
            minDist = max(10, int(0.8 * (r * scale)))  # big enough to avoid cluster of centers
            # param2 higher => fewer candidates => faster; refine radius later if you need
            circles = cv2.HoughCircles(
                roi_s, cv2.HOUGH_GRADIENT, dp, minDist,
                param1=120, param2=50, minRadius=minR_s, maxRadius=maxR_s
            )

            if circles is not None:
                # Pick best circle using gradient analysis
                cand = self._pick_best_circle(roi_s, circles)
                if cand is not None:
                    x, y, rad = cand
                    if scale != 1.0:
                        x, y, rad = x/scale, y/scale, rad/scale
                    return PipeRim(x0 + x, y0 + y, rad, confidence=0.9, is_ellipse=False)

        # ---- 2) Fallback: coarse full-frame detection (first frame or lost track)
        # downscale aggressively, detect once, scale back, then refine elsewhere
        scale = 0.5 if (h*w > 900*900) else 1.0
        if scale != 1.0:
            g_s = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            # Ignore very small rings that often come from shadows
            minR_s = max(self._min_radius, int(0.30 * min(h, w) * scale))
            maxR_s = max(minR_s + 2, int(self._max_radius * scale))
        else:
            g_s = g
            minR_s, maxR_s = self._min_radius, self._max_radius

        g_s = cv2.medianBlur(g_s, 5)
        dp = 1.5                               # smaller accumulator -> faster
        minDist = max(20, int(0.4 * min(h, w) * scale))  # avoid many candidates
        circles = cv2.HoughCircles(
            g_s, cv2.HOUGH_GRADIENT, dp, minDist,
            param1=120, param2=55, minRadius=minR_s, maxRadius=maxR_s
        )
        if circles is None:
            return None

        # Pick best circle using gradient analysis
        cand = self._pick_best_circle(g_s, circles)
        if cand is None:
            return None

        x, y, r = cand
        if scale != 1.0:
            x, y, r = x/scale, y/scale, r/scale
        return PipeRim(x, y, r, confidence=0.8, is_ellipse=False)

    def _pick_best_circle(self, gray, circles):
        """Post-filter Hough candidates to reject inner rings and select best pipe rim."""
        # gray: uint8; circles: (1, N, 3) from Hough
        g = gray
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy) + 1e-6

        h, w = g.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]

        best = None
        best_score = -1e9

        for (cx, cy, r) in circles[0]:
            # annulus band around candidate circle
            band = np.abs(np.sqrt((xx-cx)**2 + (yy-cy)**2) - r) < (0.06*r)

            # enough edge support?
            edge_band = (mag > np.percentile(mag, 80)) & band
            if edge_band.sum() < 0.02*np.pi*r:   # minimal support
                continue

            # gradient should point OUTWARD
            dx, dy = (xx-cx), (yy-cy)
            dist = np.sqrt(dx*dx + dy*dy) + 1e-6
            cosang = (gx*dx + gy*dy) / (mag*dist)
            outward = np.clip(cosang, -1, 1)
            outward_ratio = float((outward[edge_band] > 0.3).mean())

            # angular coverage (% of angles having edges)
            ang = (np.arctan2(yy-cy, xx-cx) + np.pi) % (2*np.pi)
            bins = 72
            has = np.bincount(
                np.floor(ang[edge_band]/(2*np.pi)*bins).astype(np.int32),
                minlength=bins
            ) > 0
            coverage = has.mean()

            # contrast outside vs inside (rim should be wall edge)
            t = int(0.04*r)
            inside = np.clip(r - t, 1, None)
            outside = np.clip(r + t, 1, None)
            inner_mask = np.sqrt((xx-cx)**2 + (yy-cy)**2) < inside
            outer_mask = np.sqrt((xx-cx)**2 + (yy-cy)**2) > outside
            # sample near the band to be local
            local = (np.sqrt((xx-cx)**2 + (yy-cy)**2) < r+3*t) & (np.sqrt((xx-cx)**2 + (yy-cy)**2) > r-3*t)
            inner_mean = g[inner_mask & local].mean() if (inner_mask & local).any() else 0
            outer_mean = g[outer_mask & local].mean() if (outer_mask & local).any() else 0
            contrast = float(outer_mean - inner_mean)  # rim usually brighter outside

            # prefer OUTER ring: reward radius
            score = (2.0*outward_ratio) + (1.5*coverage) + (0.01*contrast) + (0.002*r)

            # Practical guardrails for real-world robustness
            h, w = g.shape[:2]

            # Border margin: avoid giant border circles
            border_margin = 0.95 * min(cx, cy, w-cx, h-cy)
            if r >= border_margin:
                continue

            # Contrast sign: outer must be brighter than inner by at least 8
            if contrast <= 8:
                continue

            # Angle coverage weighting: stricter threshold for occluded scenes
            if coverage < 0.45:
                continue

            if outward_ratio < 0.55:
                continue  # reject inner/shadow rings

            if score > best_score:
                best_score = score
                best = (float(cx), float(cy), float(r))

        return best  # or None

    def _frst_center(self, gray, radii=(7, 9, 11), alpha=2.0, thresh=85):
        """Fast Radial Symmetry Transform for center detection (~40 lines)."""
        g = gray if gray.dtype==np.uint8 else cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        T = np.percentile(mag, thresh)
        edgemask = mag > T

        h,w = g.shape
        O = np.zeros((h,w), np.float32)   # orientation votes
        M = np.zeros((h,w), np.float32)   # magnitude votes

        ys, xs = np.where(edgemask)
        vx = gx[ys, xs] / (mag[ys, xs] + 1e-6)
        vy = gy[ys, xs] / (mag[ys, xs] + 1e-6)

        for r in radii:
            cx = (xs - r*vx).round().astype(np.int32)
            cy = (ys - r*vy).round().astype(np.int32)
            keep = (cx>=0)&(cx<w)&(cy>=0)&(cy<h)
            cx, cy = cx[keep], cy[keep]
            M[cy, cx] += 1.0
            O[cy, cx] += 1.0

        S = (M * (O ** alpha)).astype(np.float32)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(S)
        return float(maxLoc[0]), float(maxLoc[1])

    def _radius_from_polar_robust(self, gray, cx, cy, r_min, r_max):
        g = gray if gray.dtype == np.uint8 else cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        R = int(r_max)
        if R <= 0:
            return max(r_min, 1.0)

        polar = cv2.warpPolar(g, (360, R), (cx, cy), R, cv2.WARP_POLAR_LINEAR)
        dr = cv2.Sobel(polar, cv2.CV_32F, 1, 0, ksize=1)
        dr_pos = np.maximum(dr, 0)

        r0 = max(int(r_min), 1)
        band = dr_pos[r0:R, :]                       # [R', 360]
        idx = np.argmax(band, axis=0)
        rad = (r0 + idx).astype(np.int32)            # per-angle peak radii
        stren = band[idx, np.arange(band.shape[1])]

        # keep only strong angles
        thr = np.percentile(stren, 70.0)
        rad_v = rad[stren >= thr]
        if rad_v.size < 50:
            return float(np.median(rad))

        # ---- NEW: find the dominant radius mode and ignore outliers
        bins = max(32, min(128, int((r_max - r_min) / 2)))
        hist, bin_edges = np.histogram(rad_v, bins=bins, range=(r_min, r_max))
        mode_idx = int(np.argmax(hist))
        r_mode = 0.5 * (bin_edges[mode_idx] + bin_edges[mode_idx+1])

        tol = max(4.0, 0.04 * r_mode)                # ±4 px or ±4%
        inliers = np.abs(rad_v - r_mode) <= tol
        if inliers.sum() < 30:
            return float(np.median(rad_v))           # fallback

        return float(np.median(rad_v[inliers]))      # robust center of the mode

    def _radius_from_polar(self, gray, cx, cy, r_min, r_max):
        """Primary radius from polar (hardened version)."""
        R = int(r_max)
        polar = cv2.warpPolar(gray, (360, R), (cx, cy), R, cv2.WARP_POLAR_LINEAR)
        g = cv2.Sobel(polar, cv2.CV_32F, 1, 0, ksize=1)  # derivative along radius
        prof = np.mean(np.abs(g), axis=1)                # average over angles
        prof[:int(r_min*0.9)] = 0                        # forbid too small radii
        prof = cv2.blur(prof.reshape(-1,1), (9,1)).ravel()
        r = int(np.argmax(prof))
        return float(r)

    def _validate_circle_candidate(self, gray, cx, cy, r):
        """Validate single circle candidate using same metrics as _pick_best_circle."""
        g = gray
        if g.dtype != np.uint8:
            g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy) + 1e-6

        h, w = g.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]

        # annulus band around candidate circle
        band = np.abs(np.sqrt((xx-cx)**2 + (yy-cy)**2) - r) < (0.06*r)

        # enough edge support?
        edge_band = (mag > np.percentile(mag, 80)) & band
        if edge_band.sum() < 0.02*np.pi*r:   # minimal support
            return False

        # gradient should point OUTWARD
        dx, dy = (xx-cx), (yy-cy)
        dist = np.sqrt(dx*dx + dy*dy) + 1e-6
        cosang = (gx*dx + gy*dy) / (mag*dist)
        outward = np.clip(cosang, -1, 1)
        outward_ratio = float((outward[edge_band] > 0.3).mean())

        # angular coverage (% of angles having edges)
        ang = (np.arctan2(yy-cy, xx-cx) + np.pi) % (2*np.pi)
        bins = 72
        has = np.bincount(
            np.floor(ang[edge_band]/(2*np.pi)*bins).astype(np.int32),
            minlength=bins
        ) > 0
        coverage = has.mean()

        # contrast outside vs inside (rim should be wall edge)
        t = int(0.04*r)
        # VERY THIN local window to avoid picking up distant dark borders
        tn = max(2, int(0.02 * r))                       # 2% of radius (min 2 px)
        dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)

        inner_mask = (dist < (r - tn))
        outer_mask = (dist > (r + tn))
        local = (dist > (r - 3*tn)) & (dist < (r + 3*tn))  # narrow ±6% band, not 12%

        inner_mean = g[inner_mask & local].mean() if (inner_mask & local).any() else 0
        outer_mean = g[outer_mask & local].mean() if (outer_mask & local).any() else 0
        contrast = float(outer_mean - inner_mean)

        # Practical guardrails for real-world robustness
        h, w = g.shape[:2]

        # Border margin: avoid giant border circles
        border_margin = 0.95 * min(cx, cy, w-cx, h-cy)
        if r >= border_margin:
            return False

        # Contrast sign: outer must be brighter than inner by at least 8
        if contrast <= 8:
            return False

        # Angle coverage weighting: stricter threshold for occluded scenes
        if coverage < 0.45:
            return False

        # Second-derivative thinness test: require median FWHM ≤ 6 px
        try:
            polar = cv2.warpPolar(g, (360, int(r)), (cx, cy), int(r), cv2.WARP_POLAR_LINEAR)
            dr = cv2.Sobel(polar, cv2.CV_32F, 1, 0, ksize=3)
            d2r = cv2.Sobel(dr, cv2.CV_32F, 1, 0, ksize=3)  # second derivative
            # Find FWHM at each angle
            fwhm_vals = []
            for theta in range(360):
                prof = d2r[theta, :]
                if prof.size < 10:
                    continue
                peak_idx = np.argmax(np.abs(prof))
                half_max = np.abs(prof[peak_idx]) / 2
                left = np.where(prof[:peak_idx] >= half_max)[0]
                right = np.where(prof[peak_idx:] >= half_max)[0]
                if left.size > 0 and right.size > 0:
                    width = (peak_idx + right[0]) - left[-1]
                    fwhm_vals.append(width)
            if fwhm_vals and np.median(fwhm_vals) > 6.0:
                return False  # too thick/smear from glare
        except:
            pass  # skip if polar fails

        return outward_ratio >= 0.55

    def _refine_circle_fit(
        self, edges: np.ndarray, init_cx: float, init_cy: float, init_r: float
    ) -> Optional[PipeRim]:
        """Refine circle fit using edge points near initial estimate."""
        edge_points = np.column_stack(np.where(edges > 0))
        
        if len(edge_points) < 10:
            return None
        
        # Filter edge points near the initial circle
        y_coords, x_coords = edge_points[:, 0], edge_points[:, 1]
        distances = np.sqrt((x_coords - init_cx)**2 + (y_coords - init_cy)**2)
        near_circle = np.abs(distances - init_r) < (init_r * 0.15)
        
        if np.sum(near_circle) < 10:
            return None
        
        filtered_points = edge_points[near_circle]
        
        # Least squares circle fit
        refined = self._least_squares_circle_fit(filtered_points)
        
        if refined is None:
            return None
        
        cx, cy, r = refined
        
        # Validate radius
        if r < self._min_radius or r > self._max_radius:
            return None
        
        return PipeRim(cx, cy, r, confidence=1.0)
    
    def _least_squares_circle_fit(self, points: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Fit circle using least squares (algebraic fit)."""
        if len(points) < 3:
            return None
        
        y_coords, x_coords = points[:, 0], points[:, 1]
        
        # Build system: A * [cx, cy, R^2 - cx^2 - cy^2] = b
        A = np.column_stack([2 * x_coords, 2 * y_coords, np.ones_like(x_coords)])
        b = x_coords**2 + y_coords**2
        
        try:
            params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy, c = params
            r = np.sqrt(c + cx**2 + cy**2)
            return (float(cx), float(cy), float(r))
        except np.linalg.LinAlgError:
            return None
    
    def _ransac_circle_fit(self, points: np.ndarray) -> Optional[PipeRim]:
        """Fit circle to points using RANSAC."""
        best_inliers = 0
        best_circle = None
        
        for _ in range(self._ransac_iterations):
            # Sample 3 random points
            if len(points) < 3:
                return None
            
            idx = np.random.choice(len(points), 3, replace=False)
            sample = points[idx]
            
            # Fit circle to 3 points
            circle = self._fit_circle_3points(sample)
            
            if circle is None:
                continue
            
            cx, cy, r = circle
            
            # Radius constraint
            if r < self._min_radius or r > self._max_radius:
                continue
            
            # Count inliers
            distances = np.sqrt((points[:, 0] - cy)**2 + (points[:, 1] - cx)**2)
            inliers = np.abs(distances - r) < self._ransac_threshold
            num_inliers = np.sum(inliers)
            
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_circle = circle
        
        if best_circle is None:
            return None
        
        cx, cy, r = best_circle
        confidence = min(1.0, best_inliers / (len(points) * 0.1))
        
        return PipeRim(cx, cy, r, confidence)
    
    def _fit_circle_3points(self, points: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Fit circle to 3 points. Returns (cx, cy, radius)."""
        if len(points) != 3:
            return None
        
        # Convert to homogeneous coordinates (y, x format from edge detection)
        y1, x1 = points[0]
        y2, x2 = points[1]
        y3, x3 = points[2]
        
        # Compute circle center
        A = x1 - x2
        B = y1 - y2
        C = x1 - x3
        D = y1 - y3
        E = ((x1**2 - x2**2) + (y1**2 - y2**2)) / 2
        F = ((x1**2 - x3**2) + (y1**2 - y3**2)) / 2
        
        denom = (A * D - B * C)
        if abs(denom) < 1e-6:
            return None
        
        cx = (E * D - B * F) / denom
        cy = (A * F - E * C) / denom
        r = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        
        return (cx, cy, r)
    
    def _smooth_temporal(self, current: PipeRim, previous: PipeRim) -> PipeRim:
        """Apply exponential moving average to rim parameters."""
        alpha = self._temporal_alpha
        
        # Smooth base parameters
        smoothed = PipeRim(
            center_x=alpha * previous.center_x + (1 - alpha) * current.center_x,
            center_y=alpha * previous.center_y + (1 - alpha) * current.center_y,
            radius=alpha * previous.radius + (1 - alpha) * current.radius,
            confidence=current.confidence,
            is_ellipse=current.is_ellipse
        )
        
        # Smooth ellipse parameters if applicable
        if current.is_ellipse and previous.is_ellipse:
            smoothed.major_axis = alpha * (previous.major_axis or 0) + (1 - alpha) * (current.major_axis or 0)
            smoothed.minor_axis = alpha * (previous.minor_axis or 0) + (1 - alpha) * (current.minor_axis or 0)
            smoothed.angle = alpha * (previous.angle or 0) + (1 - alpha) * (current.angle or 0)
        else:
            smoothed.major_axis = current.major_axis
            smoothed.minor_axis = current.minor_axis
            smoothed.angle = current.angle
        
        return smoothed


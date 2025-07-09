import itertools
import cv2
import numpy as np
from numpy.random.mtrand import Sequence

__ARROW_PATH = "arrows.png"

def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

def point_line_distance(point, line):
    px, py = point
    vx, vy, x0, y0 = line
    numerator = abs(vy * (px - x0) - vx * (py - y0))
    denominator = np.sqrt(vx**2 + vy**2)
    return numerator / denominator

def colinearity(pts):
    pts = np.array(pts, dtype=np.float32)
    [dir_x, dir_y, point_x, point_y] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.0, 0.01)
    center = np.mean(pts, axis=0)

    total_error = 0.0
    for pt in pts:
        total_error += point_line_distance(pt[0], (dir_x, dir_y, point_x, point_y))[0]

    fits[abs(total_error)] = (dir_x, dir_y, point_x, point_y, tuple(center[0]))
    points[abs(total_error)] = pts

def line_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def find_largest_contour(contours):
    largest_contour = None
    best_size = 0.0
    for i, contour in enumerate(contours):
        if i == 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) != 7:
            continue

        # mask = np.zeros_like(grayScaleArrow)
        # cv2.drawContours(mask, [approx], 0, (0, 0, 0), 1)
        # masked_pixels = cv2.bitwise_and(grayScaleArrow, grayScaleArrow, mask=mask)
        # show_image(masked_pixels)
        # if np.all(masked_pixels[mask == 255] > 0):  # you can adjust "< 10" threshold if needed
        #     continue

        if cv2.contourArea(contour) <= best_size:
            continue

        largest_contour = contour
        best_size = cv2.contourArea(contour)

    return largest_contour

if __name__ == "__main__":
    arrowImage = cv2.imread(__ARROW_PATH)
    arrowImage = cv2.resize(arrowImage, (1280, 720))

    grayScaleArrow = cv2.cvtColor(arrowImage, cv2.COLOR_RGB2GRAY)
    #show_image(grayScaleArrow)

    _, threshold = cv2.threshold(grayScaleArrow, 40, 255, cv2.THRESH_BINARY)
    #show_image(threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = find_largest_contour(contours)
    approx = cv2.approxPolyDP(largest_contour, 0.01 * cv2.arcLength(largest_contour, True), True)
    mask = np.zeros_like(grayScaleArrow)
    cv2.drawContours(mask, contours, -1, (255, 0, 0), 1)
    masked_pixels = cv2.bitwise_and(grayScaleArrow, grayScaleArrow, mask=mask)
    show_image(mask)
    for j, point in enumerate(approx):
        x, y = point[0]
        cv2.circle(arrowImage, (x, y), 1, (0, 0, 255), 3)  # red dot of radius 1

    fits = {}
    points = {}
    for combo in itertools.combinations(approx, 4):
        colinearity(combo)

    min_error = min(fits.keys())
    h, w = arrowImage.shape[:2]
    (direction_x, direction_y, line_point_x, line_point_y, (centerx, centery)) = fits[min_error]

    min_diff = 6969696969.0
    best_pt = None
    existing_points = [tuple(p[0]) for p in points[min_error]]
    for pt in approx:
        loc_dist = point_line_distance(pt[0], (direction_x, direction_y, line_point_x, line_point_y))
        lol = loc_dist[0]
        if loc_dist[0] < min_diff and tuple(pt[0]) not in existing_points:
            min_diff = loc_dist[0]
            best_pt = pt[0]

    final_angle = line_angle((int(centerx), int(centery)), best_pt)
    M = cv2.moments(approx)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    cv2.drawContours(arrowImage, [approx], 0, (255, 0, 0), 3)
    cv2.putText(arrowImage, str(final_angle), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    show_image(arrowImage)

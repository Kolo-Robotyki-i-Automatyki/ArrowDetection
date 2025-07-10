import itertools
import cv2
import numpy as np


__ARROW_PATH = "a.jpg"
__ANGLE_THRESHOLD = 50.0


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


def colinearity(pts, fits, points):
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


def closest_point_to_line(points, line):
    closest_point = None
    min_distance = float('inf')
    for point in points:
        distance = point_line_distance(point, line)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    return closest_point


def find_best_contour(contours):
    best_contour = None
    best_size = 0.0
    best_angle = 0.0
    for i, contour in enumerate(contours):
        if i == 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) != 7:
            continue
        elif cv2.contourArea(contour) <= best_size:
            continue

        fits = {}
        points = {}
        for combo in itertools.combinations(approx, 4):
            colinearity(combo, fits, points)

        min_error = min(fits.keys())
        (direction_x, direction_y, line_point_x, line_point_y, (centerx, centery)) = fits[min_error]
        approx_points = set(tuple(p[0]) for p in approx)
        base_points = set(tuple(map(int, p[0])) for p in points[min_error])
        non_colinear_points = list(approx_points - base_points)

        closest_point = closest_point_to_line(non_colinear_points, (direction_x, direction_y, line_point_x, line_point_y))

        best_contour = contour
        best_size = cv2.contourArea(contour)
        best_angle = line_angle((int(centerx), int(centery)), closest_point)

    return best_contour, best_angle


if __name__ == "__main__":
    arrowImage = cv2.imread(__ARROW_PATH)
    arrowImage = cv2.resize(arrowImage, (1280, 720))

    grayScaleArrow = cv2.cvtColor(arrowImage, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(grayScaleArrow, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bc, ba = find_best_contour(contours)

    approx = cv2.approxPolyDP(bc, 0.01 * cv2.arcLength(bc, True), True)
    M = cv2.moments(approx)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    cv2.drawContours(arrowImage, [approx], 0, (255, 0, 0), 3)
    cv2.putText(arrowImage, str(ba), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    show_image(arrowImage)

import itertools
import cv2
import numpy as np


__ARROW_PATH = "testImages/c.jpg"
__ANGLE_THRESHOLD = 20000000.0 #50.0
__BOX_FIELD_THRESHOLD = 500.0
__ARROW_FIELD_THRESHOLD = 200.0
__ANGLES = [90.0, 90.0, 270.0, 60.0, 60.0, 60.0, 270.0]
__RIGHT_ANGLE = 90.0


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


def closest_point_to_line(points, line):
    closest_point = None
    min_distance = float('inf')
    for point in points:
        distance = point_line_distance(point, line)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    return closest_point


def line_dir(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return 0, 0
    return dx / length, dy / length


def eval_permuation(points):
    error = 0.0
    for i in range(len(points)):
        dir_ab = line_dir(points[i+1], points[(i + 2) % len(points)])
        dir_ac = line_dir(points[i], points[(i + 1) % len(points)])

        angle = np.atan2(dir_ac[1], dir_ac[0]) - np.atan2(dir_ab[1], dir_ab[0])
        error += __ANGLES[i] - np.degrees(angle)

    return error


def find_candidates(contours):
    candidates = []
    for c in contours:
        app = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        if (len(app) != 4):
            continue

        lenA = np.sqrt((app[1][0][0] - app[0][0][0]) ** 2 + (app[1][0][1] - app[0][0][1]) ** 2)
        lenB = np.sqrt((app[3][0][0] - app[2][0][0]) ** 2 + (app[3][0][1] - app[2][0][1]) ** 2)
        lenC = np.sqrt((app[2][0][0] - app[1][0][0]) ** 2 + (app[2][0][1] - app[1][0][1]) ** 2)
        maxLen = max(lenA, lenB)
        diff = abs(lenA - lenB) / maxLen
        # if diff > 0.1:
        #    continue
        if (lenB * lenC < __BOX_FIELD_THRESHOLD):
            continue
        candidates.append(c)
    return candidates


def extract_rois(image, contours):
    rois = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y+h, x:x+w]  # crop the ROI
        rois.append(roi)
    return rois


def angle_between(v1, v2):
    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    angle = (angle + np.pi) % (2 * np.pi) - np.pi  # normalize to [-π, π]
    return np.degrees(abs(angle))


def eval_group(group):
    dir_ab = group[1] - group[0]
    dir_bc = group[2] - group[1]
    dir_cd = group[3] - group[2]
    dir_da = group[0] - group[3]

    angle_abc = angle_between(dir_ab, dir_bc)
    angle_bcd = angle_between(dir_bc, dir_cd)
    angle_cda = angle_between(dir_cd, dir_da)
    angle_dab = angle_between(dir_da, dir_ab)

    return (__RIGHT_ANGLE - angle_abc) ** 2 + (__RIGHT_ANGLE - angle_bcd) ** 2 + (__RIGHT_ANGLE - angle_cda)** 2 + (__RIGHT_ANGLE - angle_dab) ** 2


def arrow_angle(pts, p1, p2):
    if len(pts) != 3:
        raise "Ugabuga, not enough points for arrow angle calculation"

    line = cv2.fitLine(np.array([p1, p2]), cv2.DIST_L2, 0, 0.0, 0.01)
    closest_point = closest_point_to_line(pts, line)
    mid_point = (p1 + p2) / 2.0
    direction_vector = closest_point - mid_point

    angle_rad = np.arctan2(direction_vector[1], direction_vector[0])
    angle_deg = np.degrees(angle_rad)

    if angle_deg < 0:
        angle_deg += 360
    return angle_deg


def colinearity(pts, fits, points):
    pts = np.array(pts, dtype=np.float32)
    [dir_x, dir_y, point_x, point_y] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.0, 0.01)
    center = np.mean(pts, axis=0)

    total_error = 0.0
    for pt in pts:
        total_error += point_line_distance(pt, (dir_x, dir_y, point_x, point_y))[0]

    fits[abs(total_error)] = (dir_x, dir_y, point_x, point_y, tuple(center))
    points[abs(total_error)] = pts


def line_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def find_best_contour(contours):
    best_contour = None
    best_angle = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < __ARROW_FIELD_THRESHOLD:
            continue
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) != 7:
            continue
        cv2.drawContours(arrowImage, [contour], -1, (255, 0, 0), 3)
        fixed_approx = [approx[j][0] for j in range(7)]

        fits = {}
        points = {}
        for combo in itertools.combinations(fixed_approx, 4):
            colinearity(combo, fits, points)

        min_error = min(fits.keys())
        (direction_x, direction_y, line_point_x, line_point_y, (centerx, centery)) = fits[min_error]
        approx_points = set(tuple(p) for p in fixed_approx)
        base_points = set(tuple(map(int, p)) for p in points[min_error])
        non_colinear_points = list(approx_points - base_points)

        closest_point = closest_point_to_line(non_colinear_points,
                                              (direction_x, direction_y, line_point_x, line_point_y))

        best_contour = contour
        best_size = cv2.contourArea(contour)
        best_angle = line_angle((int(centerx), int(centery)), closest_point)

    return best_contour, best_angle


def evaluate_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return find_best_contour(contours)


if __name__ == "__main__":
    arrowImage = cv2.imread(__ARROW_PATH)
    arrowImage = cv2.resize(arrowImage, (1280, 720))

    grayScaleArrow = cv2.cvtColor(arrowImage, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(grayScaleArrow, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = find_candidates(contours)
    rois = extract_rois(arrowImage, candidates)
    biggest_square_size = 0.0

    best_contour = None
    for roi in rois:
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            continue
        bc, ba = evaluate_roi(roi)
        if bc is not None and ba:
            if (roi.shape[0] * roi.shape[1]) < biggest_square_size:
                continue
            biggest_square_size = roi.shape[0] * roi.shape[1]
            best_contour = bc

    if best_contour is not None:
        cv2.drawContours(arrowImage, [best_contour], -1, (0, 255, 0), 3)
        # Print the angle of the best contour
        print(ba)
    else:
        raise Exception("No suitable contour found")
    show_image(arrowImage)

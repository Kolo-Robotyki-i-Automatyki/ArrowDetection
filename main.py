import itertools
import cv2
import numpy as np


__ARROW_PATH = "testImages/a.jpg"
__ANGLE_THRESHOLD = 50.0
__ANGLES = [90.0, 90.0, 270.0, 60.0, 60.0, 60.0, 270.0]


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

def line_dir(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return (0, 0)
    return (dx / length, dy / length)

def eval_permuation(points):
    error = 0.0
    for i in range(len(points)):
        dir_ab = line_dir(points[i+1], points[(i + 2) % len(points)])
        dir_ac = line_dir(points[i], points[(i + 1) % len(points)])

        angle = np.atan2(dir_ac[1], dir_ac[0]) - np.atan2(dir_ab[1], dir_ab[0])
        error += __ANGLES[i] - np.degrees(angle)

    return error

# def find_best_contour(contours):
#     best_contour = None
#     best_angle = 0.0
#     best_error = float('inf')
#     best_perm = None
#     for i, contour in enumerate(contours):
#         if i == 0:
#             continue
#
#         approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
#         if len(approx) != 7:
#             continue
#
#         for i in range(len(approx)):
#             tmp = []
#             for j in range(len(approx)):
#                 tmp.append((approx[i + j % len(approx)]))
#             err = eval_permuation(tmp)
#             if abs(err) < best_error:
#                 best_error = abs(err)
#                 best_contour = contour
#                 best_angle = err
#                 best_perm = tmp
#
#         return best_contour, best_angle
#
#     return best_contour

def find_candidates(contours):
    candidates = []
    for c in contours:
        app = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        if (len(app) != 4):
            continue

        lenA = np.sqrt((app[1][0][0] - app[0][0][0]) ** 2 + (app[1][0][1] - app[0][0][1]) ** 2)
        lenB = np.sqrt((app[3][0][0] - app[3][0][0]) ** 2 + (app[3][0][1] - app[2][0][1]) ** 2)
        maxLen = max(lenA, lenB)
        diff = abs(lenA - lenB) / maxLen
        # if diff > 0.1:
        #    continue
        candidates.append(c)
        cv2.drawContours(arrowImage, [app], 0, (255, 0, 0), 3)
    return candidates

def extract_rois(image, contours):
    rois = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y+h, x:x+w]  # crop the ROI
        rois.append(roi)
    return rois

def eval_group(group):
    dir_ab = group[1] - group[0]
    dir_cd = group[3] - group[2]
    return np.arctan2(dir_cd[1], dir_cd[0]) - np.arctan2(dir_ab[1], dir_ab[0])

def find_best_contour(contours):
    best_contour = None
    best_angle = 0.0
    best_error = float('inf')
    best_perm = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) != 7:
            continue

        for i in range(7):
            group = [approx[(i + j) % 7][0][0] for j in range(4)]
            eval_group(group)

        for i in range(len(approx)):
            tmp = []
            for j in range(len(approx)):
                tmp.append((approx[i + j % len(approx)][0]))

            error = eval_permuation(tmp)
            if abs(error) < best_error:
                best_error = abs(error)
                best_contour = contour
                best_angle = error
                best_perm = tmp

    return best_contour, best_angle

def evaluate_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_contour, best_angle = find_best_contour(contours)
    if best_contour is not None:
        cv2.drawContours(roi, [best_contour], -1, (0, 255, 0), 3)
        print(f"Best angle: {best_angle}")
    else:
        print("No valid contour found.")

if __name__ == "__main__":
    arrowImage = cv2.imread(__ARROW_PATH)
    arrowImage = cv2.resize(arrowImage, (1280, 720))

    grayScaleArrow = cv2.cvtColor(arrowImage, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(grayScaleArrow, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = find_candidates(contours)
    rois = extract_rois(arrowImage, candidates)
    for roi in rois:
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            continue
        evaluate_roi(roi)

    show_image(arrowImage)

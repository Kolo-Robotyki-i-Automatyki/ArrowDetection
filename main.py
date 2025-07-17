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
        lenB = np.sqrt((app[3][0][0] - app[3][0][0]) ** 2 + (app[3][0][1] - app[2][0][1]) ** 2)
        maxLen = max(lenA, lenB)
        diff = abs(lenA - lenB) / maxLen
        # if diff > 0.1:
        #    continue
        candidates.append(c)
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


def find_best_contour(contours):
    best_contour = None
    best_angle_err = float('inf')
    best_contour_angle = 0.0

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) != 7:
            continue

        fixed_approx = [approx[j][0] for j in range(7)]
        for i in range(7):
            group = [fixed_approx[(i + j) % 7] for j in range(4)]
            angle_err = eval_group(group)
            if abs(angle_err) < best_angle_err:
                best_angle_err = angle_err
                best_contour = contour
                left_approx = [fixed_approx[(i - 1 - k) % 7] for k in range(3)]
                best_contour_angle = arrow_angle(left_approx, group[0], group[3])

    return best_contour, best_angle_err, best_contour_angle


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
        bc, bar, bca = evaluate_roi(roi)
        if bc is not None and bar < __ANGLE_THRESHOLD:
            if (roi.shape[0] * roi.shape[1]) < biggest_square_size:
                continue
            biggest_square_size = roi.shape[0] * roi.shape[1]
            best_contour = bc

    if best_contour is not None:
        cv2.drawContours(arrowImage, [best_contour], -1, (0, 255, 0), 3)
        # Print the angle of the best contour
        print(bca)
    show_image(arrowImage)

# Arrow Detection PoC for Rover

This file contains the proof-of-concept (PoC) for the arrow detection component used in the rover's vision system.

## Heuristics Used

The following simple heuristics are applied to detect arrows:

1. The arrow should have **7 detected points** in the contour approximation.
2. **4 of those points** should be collinear (representing the base of the arrowhead).
3. Among candidates, we select **the one with the biggest area**.

---

## Approaches Tried/Thought of (But Didn't Work)

1. **Parallel Lines**  
   While some parts of the arrow should be parallel, OpenCV's contour approximation (`cv2.approxPolyDP`) isn't precise enough to reliably detect or distinguish them from unrelated shapes.

2. **Line Length Relationships**  
   Attempting to detect arrows based on ratios or absolute lengths between lines proved unreliable:
   - Varying angles and image skew distort proportions.
   - Hardcoding these relationships makes testing on varied inputs difficult.

---

## Ideas for Improved Detection

1. **Two-Stage Detection**  
   First detect **white squares**, then search for **arrows within them**. From all candidates, pick the largest arrow.

2. **Lightweight CNN**  
   Incorporate a small convolutional neural network for better robustness and generalization.

3. **Better Test Images**  
   Improve dataset quality with more **task-specific and diverse test images** to evaluate detection under real conditions.

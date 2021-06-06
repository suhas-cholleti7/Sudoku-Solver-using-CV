"""
Suhas Cholleti
Karteek Paladugu
"""


import cv2
import numpy as np
import operator
from keras.models import load_model
import SolveSudoku as sol
import copy


# Classifier used predict the numbers from an image
classifier = load_model("./digit_model.h5")

# Expected dimensions of magrins, cells and the grid in the sudoku
margin = 4
cell = 28 + 2 * margin
grid_size = 9 * cell

# Initializing video capture and video writer
cap = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
flag = 0
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1080, 620))


while True:

    ret, frame = cap.read()
    # Each frame is read, and then a adaptive thresholding is applied to convert the image to a binary image.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    threshold = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    # cv2.imshow("threshold", threshold)
    # Find the contours in the binary image
    contours, hierarchy = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_grid = None
    maxArea = 0

    # Find the biggest contour, as that would be the sudoko puzzle
    for c in contours:
        area = cv2.contourArea(c)
        if area > 25000:
            peri = cv2.arcLength(c, True)
            polygon = cv2.approxPolyDP(c, 0.01 * peri, True)
            if area > maxArea and len(polygon) == 4:
                contour_grid = polygon
                maxArea = area

    # If a contour large enough isnt present we continue to the next frame.
    # IF there is one, we the image is warped to get an image that as close to a square as possible.
    if contour_grid is not None:
        cv2.drawContours(frame, [contour_grid], 0, (0, 255, 0), 2)
        points = np.vstack(contour_grid).squeeze()
        points = sorted(points, key=operator.itemgetter(1))
        if points[0][0] < points[1][0]:
            if points[3][0] < points[2][0]:
                pts1 = np.float32([points[0], points[1], points[3], points[2]])
            else:
                pts1 = np.float32([points[0], points[1], points[2], points[3]])
        else:
            if points[3][0] < points[2][0]:
                pts1 = np.float32([points[1], points[0], points[3], points[2]])
            else:
                pts1 = np.float32([points[1], points[0], points[2], points[3]])
        pts2 = np.float32([[0, 0], [grid_size, 0], [0, grid_size], [
                          grid_size, grid_size]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, M, (grid_size, grid_size))
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = cv2.adaptiveThreshold(
            warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)

        # If flag is 0, the puzzle is not solved yet. So we solve the puzzle by first predicting the digits in the image
        # and passing those numbers to the sudoku solver.
        # If flag is 1, then it skips the solving of the sukodu as it is already solved
        # cv2.imshow("warped", warped)
        if flag == 0:

            grid_text = []
            for y in range(9):
                line = []
                for x in range(9):
                    y2min = y * cell + margin
                    y2max = (y + 1) * cell - margin
                    x2min = x * cell + margin
                    x2max = (x + 1) * cell - margin
                    cv2.imwrite("mat" + str(y) + str(x) + ".png",
                                warped[y2min:y2max, x2min:x2max])
                    img = warped[y2min:y2max, x2min:x2max]
                    x = img.reshape(1, 28, 28, 1)
                    if x.sum() > 10000:
                        prediction = classifier.predict_classes(x)
                        line.append(prediction[0])
                    else:
                        line.append(0)
                grid_text.append(line)
            print(grid_text)
            predicted_grid = li2 = copy.deepcopy(grid_text)
            result = sol.sudoku_solver(predicted_grid) # Results is None if there is no correct response.
        print("Result:", result)


        # If the numbers were predicted wrong, the sudoku cant be solved. In that scenario, the sudoku solver returns None
        # and skips the following step.
        # If we have a solved sudoku, the numbers are written on a prediction mask. The prediction mask is then warped
        # on the same plane of angle of the original frame and then displayed.
        if result is not None:
            flag = 1
            prediction_mask = np.zeros(
                shape=(grid_size, grid_size, 3), dtype=np.float32)
            for y in range(len(result)):
                for x in range(len(result[y])):
                    if grid_text[y][x] == 0:
                        cv2.putText(prediction_mask, "{:d}".format(result[y][x]), ((
                            x) * cell + margin + 3, (y + 1) * cell - margin - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
            M = cv2.getPerspectiveTransform(pts2, pts1)
            h, w, c = frame.shape
            # cv2.imshow("prediction_mask", prediction_mask)
            prediction_mask = cv2.warpPerspective(prediction_mask, M, (w, h))
            img2gray = cv2.cvtColor(prediction_mask, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask = mask.astype('uint8')
            mask_inv = cv2.bitwise_not(mask)
            img1_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            img2_fg = cv2.bitwise_and(prediction_mask, prediction_mask, mask=mask).astype('uint8')
            dst = cv2.add(img1_bg, img2_fg)
            dst = cv2.resize(dst, (1080, 620))
            cv2.imshow("frame", dst)
            out.write(dst)

        else:
            frame = cv2.resize(frame, (1080, 620))
            cv2.imshow("frame", frame)
            out.write(frame)

    else:
        flag = 0
        frame = cv2.resize(frame, (1080, 620))
        cv2.imshow("frame", frame)
        out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


out.release()
cap.release()
cv2.destroyAllWindows()

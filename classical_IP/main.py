import cv2
import numpy as np

def retinex_stable(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return None, None, None

    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l_float = l.astype(np.float32) + 1.0

    # -------- RETINEX --------
    log_l = np.log(l_float)

    blur = cv2.GaussianBlur(log_l, (101, 101), 0)

    retinex = log_l - blur

    # -------- CONTROL NORMALIZATION --------
    # Instead of full normalization, scale gently
    retinex = (retinex - np.mean(retinex)) * 30 + 128

    retinex = np.clip(retinex, 0, 255).astype(np.uint8)

    # -------- SHADOW MASK --------
    _, mask = cv2.threshold(
        l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (21,21), 0)

    # -------- SOFT BLENDING --------
    mask_f = mask.astype(np.float32) / 255.0

    # Reduce strength (VERY IMPORTANT)
    alpha = 0.6 # try 0.3–0.6

    l_final = (l * (1 - mask_f) + 
               (alpha * retinex + (1 - alpha) * l) * mask_f)

    l_final = l_final.astype(np.uint8)

    # Merge back
    lab_final = cv2.merge((l_final, a, b))
    result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)

    return img, mask, result


# RUN
input_path = r"C:\college\2nd YEAR_SEM-4\IP\CP\v2\input.jpg"

original, mask, output = retinex_stable(input_path)

if original is not None:
    cv2.imshow("Original", original)
    cv2.imshow("Mask", mask)
    cv2.imshow("Final Output", output)

    cv2.imwrite("retinex_final.jpg", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
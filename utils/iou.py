def calculate_iou(boxA: str, boxB: str, eps=1e-7) -> float:
    """
    boxA: 'x y w h'
    boxB: 'x y w h'
    """

    # [centerX, centerY, width, height]
    boxA = [float(n)*512 for n in boxA.split(' ')] 
    boxB = [float(n)*512 for n in boxB.split(' ')]
    
    # [(x_min, y_min), (x_max, y_max)]
    boxA = [(boxA[0] - boxA[2]/2, boxA[1] - boxA[3]/2), (boxA[0] + boxA[2]/2, boxA[1] + boxA[3]/2)]
    boxB = [(boxB[0] - boxB[2]/2, boxB[1] - boxB[3]/2), (boxB[0] + boxB[2]/2, boxB[1] + boxB[3]/2)]

    # Calculate tl and br for intersection
    x_start = max(boxA[0][0], boxB[0][0])
    y_start = max(boxA[0][1], boxB[0][1])
    x_end = min(boxA[1][0], boxB[1][0])
    y_end = min(boxA[1][1], boxB[1][1])

    # Calculate area of intersection
    intersection_area = max(0, x_end - x_start) * max(0, y_end - y_start)

    # Calculate area of both boxes
    boxA_area = (boxA[1][0] - boxA[0][0]) * (boxA[1][1] - boxA[0][1])
    boxB_area = (boxB[1][0] - boxB[0][0]) * (boxB[1][1] - boxB[0][1])

    # Calculate intersection over union
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area + eps)

    return iou
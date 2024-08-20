import cv2

# for i in range(50):
#     try:
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             print(f"Camera found at index {i}")
#             break
#     except:
#         pass


cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True: 
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

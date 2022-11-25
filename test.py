test_image=cv2.imread("images/greennb.jpeg")
resized_test_image=cv2.resize(test_image,(50,50))
resized_gray_image=cv2.cvtColor(resized_test_image, cv2.COLOR_BGR2GRAY)
resized_gray_image_255=resized_gray_image/255
cv2_imshow(resized_gray_image)
# print(test_images.shape)
reshaped_gray_image_255=np.reshape(resized_gray_image_255,(1,50,50))
print(resized_gray_image_255.shape)

prediction=model.predict(reshaped_gray_image_255)
prediction=prediction[0]
max_index=np.argmax(prediction)
if max_index==0:
  print("Ball")
else:
  print("Notebook")

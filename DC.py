dir_path='/content/DOGCAT/testing'

for i in os.listdir(dir_path):
  img=image.load_img(dir_path+'//'+i,target_size=(200,200))

  if img is None:
    print("Error: Could not read the image file.")
  else:
    img_array = np.array(img)
    test_img = cv2.resize(img_array,(200,200))
    test_input = test_img.reshape((1,200,200,3))
    plt.imshow(test_img)
    plt.show()

    val=model.predict(test_input)
    if val == 1:
      print("dog")
    else:
      print("cat")

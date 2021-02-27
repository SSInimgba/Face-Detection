
# How many faces?

You are provided with images of people at meetings, gatherings, group photos, etc. Count the number of faces you can spot in each image. There will be no more than 15 faces in each of the images. Assume that half or more of each face will be visible.

**TASKS**
 
- Given an input of A 2D grid of pixel values provided (in regular text format through STDIN) that represents the pixel-wise values from the images output the number of faces detected in the image. The input will contain two integers, R  and C, representing the number of rows and columns of image pixels, respectively. 

- Create a GitHub repository for your solution to the above problem.

- Discuss in detail how you solved the problem.

- Are there other ways the problem could be solved? Discuss briefly

# SOLUTION

Transform the input data into images represented in arrays
```
CHANNEL = 3
image_1_path = '/content/drive/MyDrive/NB/input/input00.txt' #insert your image path


def txt_to_image(image_path):
  """takes in a path to the txt file containing the image pixels and returns an image in array form"""
  with open(image_path, "r") as file:  # read the data from the text file
    lines = file.readlines()
    image_list = [[line.replace(" ", ",").split(",")] for line in lines[1:]]
  first_line = lines[0]
  first_line = first_line.split()
  R = int(first_line[0]) # rows
  C = int(first_line[1]) # columns
  image = np.array(image_list)
  image = image.reshape(R, C, CHANNEL).astype(np.uint8) 
  image = image[:,:,::-1] # transform to RGB
  return image
```
Method 1: Use machine learning via Haarcascades face detector from openCV
```
# method 1 using openCV via haarcascades
def haarcascades_facedetector(image):
  """takes in an image and applies haarcascades face detector 
  and returns number of faces detected"""
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # convert image to grayscale
  faceCascade = cv2.CascadeClassifier('/content/drive/MyDrive/NB/haarcascade_frontalface_default.xml')
  haar_faces = faceCascade.detectMultiScale(gray, 
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(20, 20))
  faces = len(haar_faces)
  return faces
```
Method 2: Use Deeplearning via MTCNN 
```
# method 2 using Deep Learning via MTCNN
def mtcnn_facedetector(image):
  """takes in an image and applies mtcnn face detector 
  and returns number of faces detected"""
  # create the detector, using default weights
  detector = MTCNN()
  mtcnn_faces = detector.detect_faces(image)
  faces = len(mtcnn_faces)
  return faces
```
USAGE

```
image = txt_to_image(image_path)
print(haarcascades_facedetector(image)) # method 1
print(mtcnn_facedetector(image)) # method 2 (chosen method)

```

Click on [Notebook](https://github.com/SSInimgba/Face-Detection/blob/main/Face_Detection_and_Count.ipynb) for detailed explanation.

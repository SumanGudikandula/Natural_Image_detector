************************************************************************************
# Natural_Image_detector
CNN model in pytorch for Natural Image detection & KTinker GUI.
************************************************************************************

#1 ......Downloading Dataset:
-------------------------------------------------------------------------------------
Download the Natural Image Dataset images of around 6k-7k images from link https://www.kaggle.com/prasunroy/natural-images.

#2 ......Downloading Program:
-------------------------------------------------------------------------------------
In the repository 'Natural_Image_detector' check for Natural_Image_Classifier.ipynb and download it.

#3 ......Setup for running Program:
-------------------------------------------------------------------------------------
Install Anaconda Jupyter in your computer. If not already present.

Open ipynb file in jupyter in chrome or Internet explorer. 

Update these two paths to your local desktop file locations... one for loading in dataset and other for saving final model.
#Images files folder
path = r'C:\Users\hp\OneDrive\Desktop\DM ALL\PROJ\archive'
data_dir = r'C:\Users\hp\OneDrive\Desktop\DM ALL\PROJ\archive\natural_images'

Now run the program: 

It builds model and prints train, test accuracies and along with sample batch predictions.
It also displays the Dataframe of actual vs Predicted image labels for all test data.

Your Pytorch model will be saved in a file at location path = r'C:\Users\hp\OneDrive\Desktop\DM ALL\PROJ\archive' with name
  model.pth
  
 
#4 .....Ktinker GUI features:
-------------------------------------------------------------------------------------

Finally your GUI of kTinker opens at the end after entire model is built.

Now upload any image from upload button... GUI displays probabilities like 
if cat image given:
Aeroplane: 0.1% Car: 2% Cat:96% etc ...

The max probability is used for detecting the final prediction, which is also displayed.
 
Instructions for installing special libraries:

#5 .....Special Notes:
-------------------------------------------------------------------------------------
When you open jupyter from Anaconda a command promt will be opened...
Install ktinker using command : pip install tk
Install pytorch using command :  pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

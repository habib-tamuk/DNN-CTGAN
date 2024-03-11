4-Class Classification Testruns for CTGAN, these are NEW Testruns created on 2/28/24
**(Updated 3/5/24 to include images of Heatmaps, ROC Curves, and ANN Training (Acc/Loss) Graph)

The code was ran (and designed) for Anaconda Spyder using Python3
Feel free to copy and paste sections for your own use if you have issues running the providing script all at once

Description: I went through each and every testrun folder and concatenated that with my real data and saved their respective results in each folder in a .txt file.

For simplicity and also to save disc space, I will not include the synthethic datasets in each folder again as they take up a lot of space, If they are needed, you can refer to the OneDrive link I sent 2/27/24 which includes similar data with each synthetic dataset included.

--------------------------------------------------------------------------------------------------------------------------------------

Note:
- The synthetic dataset for the original model was not found/never saved, so for simplicity, I also did not include it inside this folder.
- After further testing, I found out that synthetic datasets 0 and 3 are the exact same, either I accidentally saved it twice or the CTGAN generated the exact same dataset, which is possible as each dataset used the exact same parameters for training in the GAN network

--------------------------------------------------------------------------------------------------------------------------------------
Here are the contents of the zip folder (very similar to the one I have on my OneDrive):

Keras_Model - Includes the exact same ANN Model I have been using for my model (So I do not need to retrain the neural network)

src - I used this python file to run my code for every synthetic dataset one by one and record my resuls

CTGAN_Testrun0-4 - Each of these folders will have the results I got from using that specific generated synthetic dataset (from the OneDrive zip folder)
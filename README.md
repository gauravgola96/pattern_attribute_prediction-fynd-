# pattern_attribute_prediction
## Classification Report
![classification_report](https://user-images.githubusercontent.com/32256853/48228897-90faf700-e3cc-11e8-8958-339bac45a68b.png)

Problem breakdown : 

1) Data analysis and preparation:
	- Total data provided : around 3500. Out of 12 classes 11 classes had around 8% of data but the class Colourblock had only 3% of data.
	- Performed data augmentation such that I had 600 images per class (Performed more augmentation on Colourblock class)	
	
	PROBLEMS:
 	- Features of models are different.
	- Different background.
	- Some clothes were without models (again different feature).
	
	SOLUTION: 
	- Approach 1) -> LINK- https://github.com/qianghaowork/cut_api
	  Most of images have simple background and faces could be detected with good accuracy, a straightforward solution is proposed. 	  Firstly, use openCV Harr Cascades face detection. Secondly, use edges to determine the object location. 
          Thirdly, calculate naive background model and skin model to obtain the clothes and remove the background and human body

        - Approach 2) Simply cropping the center part images assuming it contains most of information about the pattern attribute   
          of the clothes.
	
	Path for approches in repo --> Augmentation>---


2) Data Selection (Approach 1 or Approach 2)
	
	- Tested the data from above two approach on VGG16 and VGG19(For Baseline).
	- Freezing first 12-15 layers.
	- Approach 2 performed better than approach 1.
	- When I observed the Approach 1 data, it has lot of images from different classes but looking exactly alike hence confusing the model.


3) Model Selection 
	- Total Data after augmentation - 7200 from 12 classes (train data ).
	- 7200 data is splited into test,train and validation datasets.
	- Performed model training on Resnet50,InceptionV3,Densenet121,Xception (https://keras.io/applications/#documentation-for-individual-models)

	- Selected the number of freezed layer such that total trainable parameters is aroun 25% of total parameters.
	- All the models got stuck at training accuracy at 83-88% and validation accuracy fluctuates between 40 to 55%.
	- Even after more than 30 epoch I was unable reach a stable validation accuracy and training accuracy more than 90% (at best).
	- Optimizer nadam (Nesterov Adam optimizer)
	- ReduceLRonplateau is used as one of the callback while model training so as to reduce learning rate if val loss is not decereasing.
	- lr = 0.001
	

	PROBLEMS:
	- Seems like data is overfitting.
	- But train data is unable to cross even 90% (after more than 30 epochs)
	- During the inital epochs the validation accuracy was more than training accuracy which is due to underfitting by chance.

	SOLUTION:
	
	- I targeted to atleast make my training accuracy should atleast cross 90% accuracy.
	- And validation accuracy should be stable since the start of training.
	- Took help from this github issues thread - (https://github.com/keras-team/keras/issues/1006)
	- I followed the same procedure as explained by @Pualraita in the github issues thread.
	- I prepared 4 subset of train dataset (7200)
	 
	-- subset_1 - data without any augmentation (around 360)
	-- subset_2 and subset_3 - data with augmentation (720 each)
	-- train - Augmentated data (around 5200)


	-----------------subset_1---------------------------
	- Load the Xception model with imagenet weights , Freezed first 50 layers.
	- 10 epoch
	- Trained the model on subset_1 data and saved the weights.
	
	---------------subset_2-----------------------------
	- Load the same network with no weights , Freezed first 80 layers (such as 30 layers still has the information stored for the data
		without any augmentation)
	- 15 epoch
	- Load weights from subset_1 model and trained it on subset_ data.
	- Saved the model weights.

	---------------subset_3------------------------------
	- Load the same network with no weights , Freezed first 100 layers (such as 20 layers still has the pre-trained information)
	- 15 epoch
	- Load weights from subset_2 model and trained it on subset_ data.
	- Saved the model weights

<<<<<<< HEAD
          --------------train data------------------------------
=======
         --------------train_data------------------------------
>>>>>>> b3a1b6250fdd9cc992659dcc0806b2e80d4659cb
	- Load the same network with no weights , Freezed first 110 layers (such as 10 layers still has the pre-trained information)
	- 30 epoch.
	- Load weights from subset_3 model and trained it on train data.
	- Saved the model weights

	Path for code in repo ->  'Model_train_fynd.ipynb'
<<<<<<< HEAD
	Path for subset model weights in repo --> 'model_subset/'
=======
	Path for subset model weights in repo --> 'model_subset/' (https://drive.google.com/drive/folders/12Zhd2LgytHEGTfnBfHQddzV0NWkoA_32?usp=sharing)
>>>>>>> b3a1b6250fdd9cc992659dcc0806b2e80d4659cb
	
	Path for Confusion matrix in repo --> 'confusion_matrix.ipynb'


	OBSERVATIONS:
 	
	- Now the training accuracy easily reaches more than 95% under 30 epoch.
	- Validation accuracy is 62%  (but stable without much of fluctutations)


4) run.py 
	
	- Made a run.py with argument parser for running the code on terminal.
<<<<<<< HEAD
	- Provide only image path and returns the prediction.
=======
	- Provide only image path and it returns the prediction.
>>>>>>> b3a1b6250fdd9cc992659dcc0806b2e80d4659cb



	
	 






	

	







	




<<<<<<< HEAD



=======
>>>>>>> b3a1b6250fdd9cc992659dcc0806b2e80d4659cb

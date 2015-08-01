## Quora Answer Classifier Challenge

Quora uses a combination of machine learning (ML) algorithms and moderation to ensure high-quality content on the site. High answer quality has helped Quora distinguish itself from other Q&A sites on the web. Your task will be to devise a classifier that is able to tell good answers from bad answers, as well as humans can.  A good answer is denoted by a +1 in our system, and a bad answer is denoted by a -1.

The dataset provided by Quora for this classification tasks consists of features abstracted into numerical quantities. So, a major part of dataset pre-processing was abstracted away. Keeping in mind the time limitations, and the fact that some of the features are extraneous, I decided to use a Random Forest for the classification task.

A sample dataset was provided, upon which a Random Forest provided an 83.6% classification accuracy on the test set. On the large real world dataset, which Quora uses for evaluation, my code achieved a score of 86.11.

## Running the simulation

The simulation code is pretty straightforward. The `rf.py` file contains the code which was submitted on the Quora Challenges page, and takes input by STDIN. The `rf_fileIO.py` file uses the text files `input00.txt` and `output00.txt` for input and the true labels of output respectively. Note that the code requires Scikit-learn and Numpy libraries.



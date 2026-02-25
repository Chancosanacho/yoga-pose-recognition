#### Yoga Pose Recognition and Correction System



This project implements a real-time yoga posture recognition and correction system based on human pose estimation and deep learning.



It combines:

\- MediaPipe for keypoint extraction

\- A deep learning model (Conv1D + GRU) for pose classification

\- An angle-based biomechanical evaluation system

\- Real-time visual and textual feedback





##### Project Structure



* references/ -> Reference images for correction



* \*.py -> Main scripts:
   *main\_realtime.py*: Main file of the application. Implements:

&nbsp;	– Real-time video capture

&nbsp;	– Keypoint extraction

&nbsp;	– Model inference

&nbsp;	– Integration of the feedback system

&nbsp;   	Acts as the execution core of the system.

&nbsp;    *pose\_angles.py*: Encapsulates the functions related to:

&nbsp;	– Keypoint extraction through MediaPipe Pose

&nbsp;	– Geometric calculation of joint angles

&nbsp;	This module abstracts the biomechanical logic from the rest of the 	system.

&nbsp;    *pose\_feedback.py*: Implements the posture correction logic:

&nbsp;	– Comparison with reference metrics

&nbsp;	– Deviation detection

&nbsp;	– Generation of corrective messages

&nbsp;	Represents the biomechanical interpretation layer.

&nbsp;    *evaluate\_pose.py*: Script intended for evaluating the performance of the 	model outside the real-time environment, allowing classification metrics 	to be analyzed in a controlled manner.

&nbsp;    *extract\_reference\_angles.py*: Script responsible for generating 	biomechanical statistics from the reference images.



* README.md -> Project description



* .gitignore -> Files ignored by Git



##### Requirements



Lenguaje: Python 3.10.11



Librerías principales:



TensorFlow: 2.19.0



Keras: 3.12.0



MediaPipe: 0.10.13



NumPy: 1.26.4



OpenCV:



opencv-python 4.9.0.80



opencv-contrib-python 4.11.0.86



SciPy: 1.15.3



Matplotlib: 3.10.8



h5py: 3.15.1



protobuf: 4.25.8



##### How to Run



Run main\_realtime.py


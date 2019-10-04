import numpy as np
from collections import Counter
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout
from keras.optimizers import Adam,Adadelta
from sklearn.cross_validation import train_test_split
from keras.models import load_model

label_tax = 4
model_type = 1 #1=AA_code;2=physics_Properties;3=merge

##################################################################
if (model_type == 1):
	#         C,G,P,W,Y,T,S,F,M,A,V,I,L,H,K,Q,N,E,D,R
	code_R = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
	code_D = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
	code_E = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
	code_N = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
	code_Q = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
	code_K = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
	code_H = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
	code_L = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
	code_I = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
	code_V = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
	code_A = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
	code_M = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
	code_F = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
	code_S = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
	code_T = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	code_Y = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	code_W = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	code_P = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	code_G = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	code_C = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	print("only amino acid code.")

matrix_size = len(code_R)
code_X = [1 for i in range(matrix_size)]
code_n = [0 for i in range(matrix_size)]

####################################################################


def aa_ref2npy(ref_Data, len_w):
	#####################################
	seq = []
	X = []  # seqs
	Y = []  # label for all seqs
	f = open(ref_Data, 'r')
	try:
		for line in f:
			line = line.strip('\n')
			if line[0] == '>':
				ll = line.split()
				Y.append(int(ll[-1]) - 1)
			else:
				X.append(line)
	finally:
		f.close()

	len_input = len(Y)
	len_input_1 = len(X)
	if len_input == len_input_1:
		print("ok! there is the same number (" + str(len_input) + ") of labels and sequences. ")
	else:
		print("error! the number of labels (" + str(len_input) + ") and the number of sequences (" + str(
			len_input) + ") are different."), exit()
	####################################
	n_size = len(X)
	print(n_size)
	#XX = []
	YYY = []
	XXX = []

	for i in range(n_size):
		seq = X[i]
		seq_label = Y[i]
		len_seq = len(X[i])  # length of seq
		pos_end = len_seq - 1  # seq end position
		pos = 0  # seq start position
		# num_s = len_seq // len_w  # number of samples in a seq
		# rem = len_seq % len_w
		# if rem >= 50:  # the rest number of seq be read
		# 	num_s = num_s + 1
		# if num_s == 0:
		# 	num_s = num_s + 1  # short seq fill with '-'
		# for j in range(1):
		XX = []
		for k in range(len_w):
			if pos <= pos_end:
				ctr = seq[pos]
			else:
				ctr = '-'
			if ctr == 'R':
				XX.append(code_R)
			elif ctr == 'D':
				XX.append(code_D)
			elif ctr == 'E':
				XX.append(code_E)
			elif ctr == 'N':
				XX.append(code_N)
			elif ctr == 'Q':
				XX.append(code_Q)
			elif ctr == 'K':
				XX.append(code_K)
			elif ctr == 'H':
				XX.append(code_H)
			elif ctr == 'L':
				XX.append(code_L)
			elif ctr == 'I':
				XX.append(code_I)
			elif ctr == 'V':
				XX.append(code_V)
			elif ctr == 'A':
				XX.append(code_A)
			elif ctr == 'M':
				XX.append(code_M)
			elif ctr == 'F':
				XX.append(code_F)
			elif ctr == 'S':
				XX.append(code_S)
			elif ctr == 'T':
				XX.append(code_T)
			elif ctr == 'Y':
				XX.append(code_Y)
			elif ctr == 'W':
				XX.append(code_W)
			elif ctr == 'P':
				XX.append(code_P)
			elif ctr == 'G':
				XX.append(code_G)
			elif ctr == 'C':
				XX.append(code_C)
			elif ctr == '-':
				XX.append(code_n)
			else:
				XX.append(code_X)
			pos = pos + 1
		XXX.append(XX)
		# XX = []
		YYY.append(seq_label)
	print(len(XXX))
	print(len(YYY))
	XXX = np.array(XXX)  # XXX.reshape(-1,1,750,20)
	YYY = np.array(YYY)
	np.save(ref_Data + ".X", XXX)
	np.save(ref_Data + ".Y", YYY)
	print("ok! windows is " + str(len_w) + ".")
	print("ok! raw data has been saved as a npy file " + ref_Data + ".X/Y")

####################################################################################

##########################################################################################

def DL_TrainTest(ref_Data, len_w):
        test_split_rate=0.2
        nb_filters = 50
        kernel_s = 3
        n_batch = 10
        n_echos = 20
        dropout1 = 0.10
        dropout2 = 0.10
        print ("now runing is DL_TrainTest.")
        print ("test_split_rate: ", test_split_rate)
        print ("nb_filters: ", nb_filters)
        print ("kernel_s: ", kernel_s)
        print ("n_batch: ", n_batch)
        print ("n_echos: ", n_echos)
        print ("dropout1: ", dropout1)
        print ("dropout2: ", dropout2)
        X = np.load(ref_Data + ".X.npy")
        Y = np.load(ref_Data + ".Y.npy")
        print ("ok! the npy file " + ref_Data + ".X/Y.npy are loaded!" )
        n_classes = 4     #len(np.unique(Y)) should be modified by the last label number of the training data
        # print(Y)
        print ("ok! all labels are in " + str(n_classes) + " kinds." )
        YY_t = []
        for i in Y:
            ll = np.zeros(n_classes)
            ll[i] = 1
            YY_t.append(ll)
        YY_t = np.array(YY_t)
        print (YY_t)
        for i in range(5):
                print("now test "+ str(i) + ".")
                X_train,X_test,Y_train,Y_test = train_test_split(X,YY_t,test_size=test_split_rate)
                X_train = X_train.reshape(-1,1,len_w,matrix_size)
                X_test = X_test.reshape(-1,1,len_w,matrix_size)
                model = Sequential()
                model.add(Conv2D(filters=nb_filters,kernel_size=(3,3),padding='same',input_shape=(1,len_w,matrix_size),data_format='channels_first'))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(3,3)))
                model.add(Dropout(dropout1))
                model.add(Flatten())
                model.add(Dense(100,activation='relu'))
                model.add(Dropout(dropout2))
                model.add(Dense(n_classes,activation='softmax'))
                model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])   #
                model.fit(X_train,Y_train,batch_size=n_batch,epochs=n_echos,verbose=1)
                model.save(ref_Data + '.h5')
                score = model.evaluate(X_test,Y_test,verbose=0)
                print("Ended: ",score[0],score[1])

##############################################################################################

# unknown_data = 'training_data.faa'
ref_Data = 'training_data.faa'
len_w = 900

if 1:
	aa_ref2npy(ref_Data=ref_Data,len_w=len_w)
if 1:
	DL_TrainTest(ref_Data=ref_Data,len_w=len_w)

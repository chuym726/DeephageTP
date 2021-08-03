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

def aa_txt2matrix(txt, len_w):
	XX = []
	YYY = []
	XXX = []

	seq = str(txt)
	len_seq = len(seq)                  # length of seq
	pos_end = len_seq - 1                # seq end position
	pos = 0                              # seq start position
	# num_s = len_seq // len_w             # number of samples in a seq
	# rem = len_seq % len_w
	# if rem >= 50 :                      # the rest number of seq be read
	#         num_s = num_s + 1
	# if num_s == 0:
	#         num_s = num_s + 1                # short seq fill with '-'
 	# for j in range(num_s):
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
	XXX = np.array(XXX)
	return XXX

#####################################################################



def predict_and_loss(model, data_to_predict, len_w):
	seq = []
	X = []  # seqs
	Y = []  # label for all seqs
	f = open(data_to_predict, 'r')
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
			len_input_1) + ") are different."), exit()
	##########################################################################
	model = load_model(model)
	print("model has loaded.")
	print("predicting!")
	for mm in range(len_input_1):
		Xm = X[mm]
		Xm = aa_txt2matrix(txt=Xm, len_w=len_w)
		Xm = Xm.reshape(-1, 1, len_w, matrix_size)
		Yp = model.predict_classes(Xm, verbose=0)
		Ypl = len(Yp)
		# print(Ypl)
		loss_bin = []
		for i in range(Ypl):
			Yp_t = []
			Ypli = Yp[i]
			ll = np.zeros(label_tax)
			ll[Ypli] = 1
			Yp_t.append(ll)
			Yp_t = np.array(Yp_t)
			Xmi = Xm[i].reshape(-1, 1, len_w, matrix_size)
			scores = model.evaluate(Xmi, Yp_t)
			loss_bin.append((Ypli, scores[0]))
		# print(loss_bin)
		# if len(loss_bin)>1:
		# loss_bin.pop()
		# loss_bin.sort(reverse = True, key=lambda x:x[1])
		# print(loss_bin.sort(reverse = True, key=lambda x:x[1]))
		minIndex = 0
		for i in range(0, len(loss_bin)):
			if loss_bin[minIndex][1] > loss_bin[i][1]:
				minIndex = i
		print("the min loss and its predicted label:", loss_bin[minIndex], "label:", loss_bin[minIndex][0], "min_loss:",
			  loss_bin[minIndex][1])
		"""
		loss_bin_dict_new = {}
		for i in loss_bin:
			loss_bin_dict_new[i[0]] = i[1]
		print(loss_bin_dict_new)
		loss_bin_dict_new = sorted(loss_bin_dict_new.items(),key=lambda item:item[1],reverse=False)	
		print(loss_bin_dict_new)
		loss_bin_dict_new_min = loss_bin_dict_new[:1]
		"""
		# loss_bin.sort(reverse = False, key=lambda x:x[1])
		# loss_dic = dict(loss_bin)
		# print(loss_dic)
		# loss_dic_sort = sorted(loss_dic.items(),key=lambda item:item[1],reverse=False)
		# print(loss_dic_sort)
		# loss_dic_sort_min = loss_dic_sort[:1]
		# print(loss_dic_sort_min)
		print("the number of splited fragment:", Ypl)
		print("the predicted labels and its losses of all fragment(sequences splited by 500AA):", loss_bin)
		print("id:", mm, " orgin label:", str(Y[mm]), " predict label:", str(loss_bin[minIndex][0]), "loss",
			  str(loss_bin[minIndex][1]))
	# print(loss_dic)
	# print(loss_dic.values)
	# min_loss_label = min(loss_dic.items(), key=lambda x: x[1])[0]
	# print(min_loss_label)
	# Yp = np.ones(Ypl, dtype=int)
	# Yp = Yp.dot(min_loss_label)
	# yll = []
	# for i in range(Ypl):
	#   ll = np.zeros(label_tax)
	#    ll[Yp[i]] = 1
	#    yll.append(ll)
	# yll = np.array(yll)
	# scores = model.evaluate(Xm,yll)
	# print("id:",mm," orgin label:",str(Y[mm])," predict label:",str(min_loss_label)," loss:",scores[0])
	print("all is ended.")


unknown_data = 'example_data.fa'
ref_Data = 'training_data.faa'
len_w = 900
if 1:
	aa_ref2npy(ref_Data=ref_Data, len_w=len_w)

if 1:
	predict_and_loss(model=str(ref_Data) + ".all.h5", data_to_predict=unknown_data, len_w=len_w)

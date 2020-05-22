import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from LSTM import LSTM
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch
from GRU import GRU

# plot confusion matrix
def plot_confusion(confu_mat):
    confu_mat = np.tril(confu_mat, k=0) # plot only lower triangular
    df_cm = pd.DataFrame(confu_mat, range(confu_mat.shape[0]), range(confu_mat.shape[1]))
    plt.figure(figsize=(20,20))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.title('confusion matrix') ; plt.savefig('confusion_matrix.png')

# load data
filename = "/home/mint/Desktop/Data_Set/Deep_Learning_NCTU_Homework/DL_HW2/Question_1/covid_19.csv"
use_cols_list = [*range(40, 85)]
seq = pd.read_csv(filename, skiprows=[0, 1], usecols=use_cols_list)

# Computing co-variance
seq_value = seq.values
for i in range(len(seq)):
    seq_num = seq_value
    seq_mat = np.array(seq_num)


# use difference matrix to compute correlation coefficient matrix
diff_mat = np.diff(seq_mat)
cov_mat = np.corrcoef(diff_mat)
col_range_list = [i for i in range(10)]
plot_confusion(cov_mat[0:10, col_range_list])

# thresh_hold = 0.8
# select_cont_list = []
# for i,c in enumerate(cov_mat):
#     if all(x>thresh_hold for x in c.astype(float)):
#         select_cont_list.append(i)

select_cont_list = []
cont_threshold = 40
thresh_hold = 0.55
for i,c in enumerate(cov_mat):
    # jg_row is the amount of corelation coef that is greater than thresh_hold
    jg_row = sum([1 if x>thresh_hold else 0 for x in c.astype(float)])
    # select the country which amount of corelation coef is greater than cont_threshold
    if jg_row > cont_threshold:
        select_cont_list.append(i)

# make the selected and not-selected country to form a matrix
select_cont_mat = np.zeros((len(select_cont_list),45))
select_cont_mat[:,:] = seq_mat[select_cont_list,:]

n_select_cont_list = []
for idx,row in enumerate(seq_mat):
    if idx not in select_cont_list:
        n_select_cont_list.append(row)
n_select_cont_mat = np.array(n_select_cont_list)


# generate training data
train_x = [] ; train_y = []
L = 7
for row in select_cont_mat:
    for idx,item in enumerate(row) :
        if(idx+L == len(row)): # exceed the row
            break
        train_x.append(np.array(row)[idx:idx+L])
        ascend = 1 if row[idx+L]-row[idx+L-1] > 0 else 0
        train_y.append(ascend)


# generate testing data
test_x = [] ; test_y = []
for row in n_select_cont_mat:
    for idx,item in enumerate(row) :
        if(idx+L == len(row)): # exceed the row
            break
        test_x.append(np.array(row)[idx:idx+L])
        ascend = 1 if row[idx+L]-row[idx+L-1] > 0 else 0
        test_y.append(ascend)

train_x_arr = np.array(train_x) ; train_y_arr = np.array(train_y)
test_x_arr = np.array(test_x)   ; test_y_arr = np.array(test_y)

train_tensor_x = torch.Tensor(train_x_arr).cuda()
train_tensor_y = torch.Tensor(train_y_arr).cuda()
test_tensor_x = torch.Tensor(test_x_arr).cuda()
test_tensor_y = torch.Tensor(test_y_arr).cuda()

LR = 0.0001
EPOCHS = 200
BATCH = 20
# TIME_STEP = (45-1)-(L-1)
TIME_STEP = 1
lstm = LSTM(in_dim=L, hid_dim=2*L, layers=5, out_state = 2).cuda()
lstm.weightInit()
gru = GRU(in_dim=L, hid_dim=2*L, layers=5, out_state = 2).cuda()

optimizer = torch.optim.Adam(lstm.parameters(), lr = LR)
loss_fun = nn.CrossEntropyLoss().cuda()

# For batch normalization , training data should be wrapped into wrapper
train_data = Data.TensorDataset(train_tensor_x,train_tensor_y)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH, shuffle=True)

train_acc_arr = np.empty((0))
test_acc_arr = np.empty((0))


for epoch in range(EPOCHS) :
    lstm.train()
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        output = lstm(inputs.view(-1,TIME_STEP,L))
        labels = labels.squeeze_().long()

        loss = loss_fun(output, labels)                         # Calculate the loss
        optimizer.zero_grad()                                   # Clear off the gradient in (w = w - gradient)
        loss.backward()                                         # Backpropagation
        optimizer.step()                                        # Update the weights
        if step % EPOCHS == 0 :
            lstm.train(False);lstm.eval()
            train_pred_y = lstm(train_tensor_x.view(-1,TIME_STEP,L))
            train_pred_y = torch.max(train_pred_y,1)[1].cpu().data.numpy().squeeze()
            test_pred_y = lstm(test_tensor_x.view(-1,TIME_STEP,L))
            test_pred_y = torch.max(test_pred_y,1)[1].cpu().data.numpy().squeeze()
            train_acc = np.mean(train_pred_y == train_tensor_y.cpu().data.numpy())
            test_acc = np.mean(test_pred_y == test_tensor_y.cpu().data.numpy())
            train_acc_arr = np.append(train_acc_arr,train_acc)
            test_acc_arr = np.append(test_acc_arr,test_acc)
            print("EPOCH : ", epoch, "training_acc : {:.2f}".format(train_acc),"testing_acc : {:.2f}".format(test_acc))
            lstm.train(True)

plt.figure();plt.plot(list(train_acc_arr));plt.title('LSTM train accuracy');plt.savefig('LSTM_train_accuracy_batch_{:f}_lr_{:f}_L_{:f}.png'.format(BATCH,LR,L))
plt.figure();plt.plot(list(test_acc_arr));plt.title('LSTM test accuracy');plt.savefig('LSTM_test_accuracy_batch_{:f}_lr_{:f}_L_{:f}.png'.format(BATCH,LR,L))
torch.save(lstm.state_dict(),'./LSTM_training_model_batch_{:f}_lr_{:f}_L_{:f}'.format(BATCH,LR,L))



# print('GRU Learning Process')
# optimizer = torch.optim.Adam(gru.parameters(), lr = LR)
# loss_fun = nn.CrossEntropyLoss()
# train_acc_arr = np.empty((0))
# test_acc_arr = np.empty((0))
# loss_list = []
# thresh_hold_count = 0
# for epoch in range(EPOCHS) :
#     # lstm.train()
#     for step, (inputs, labels) in enumerate(train_loader):
#         inputs = Variable(inputs).cuda()
#         labels = Variable(labels).cuda()
#
#         output = gru(inputs.view(-1,TIME_STEP,L))
#         labels = labels.squeeze_().long()
#
#         loss = loss_fun(output, labels)                         # Calculate the loss
#         optimizer.zero_grad()                                   # Clear off the gradient in (w = w - gradient)
#         loss.backward()                                         # Backpropagation
#         optimizer.step()                                        # Update the weights
#         if step % EPOCHS == 0 :
#             gru.train(False);gru.eval()
#             train_pred_y = gru(train_tensor_x.view(-1,TIME_STEP,L))
#             train_pred_y = torch.max(train_pred_y,1)[1].cpu().data.numpy().squeeze()
#             test_pred_y = gru(test_tensor_x.view(-1,TIME_STEP,L))
#             test_pred_y = torch.max(test_pred_y,1)[1].cpu().data.numpy().squeeze()
#             train_acc = np.mean(train_pred_y == train_tensor_y.cpu().data.numpy())
#             test_acc = np.mean(test_pred_y == test_tensor_y.cpu().data.numpy())
#             train_acc_arr = np.append(train_acc_arr,train_acc)
#             test_acc_arr = np.append(test_acc_arr,test_acc)
#             print("EPOCH : ", epoch, "training_acc : {:.2f}".format(train_acc),"testing_acc : {:.2f}".format(test_acc))
#             gru.train(True)
#
# plt.figure();plt.plot(list(train_acc_arr));plt.title('GRU train accuracy');plt.savefig('GRU_train_accuracy_batch_{:f}_lr_{:f}_L_{:f}.png'.format(BATCH,LR,L))
# plt.figure();plt.plot(list(test_acc_arr));plt.title('GRU test accuracy');plt.savefig('GRU_test_accuracy_batch_{:f}_lr_{:f}_L_{:f}.png'.format(BATCH,LR,L))
# torch.save(gru.state_dict(),'./GRU_training_model_batch_{:f}_lr_{:f}_L_{:f}'.format(BATCH,LR,L))


# # %%%%%%%%%%%%  MSE %%%%%%%%%%%%%%%%%%
# # ----------------------------------LSTM-----------------------
# lstm = LSTM(in_dim=L).cuda()
# lstm.weightInit()
# optimizer = torch.optim.Adam(lstm.parameters(), lr = LR)
# loss_fun = nn.MSELoss()
#
# # For batch normalization , training data should be wrapped into wrapper
# train_data = Data.TensorDataset(train_tensor_x,train_tensor_y)
# train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH, shuffle=True)
#
# train_acc_arr = np.empty((0))
# test_acc_arr = np.empty((0))
#
#
# loss_list = []
# thresh_hold_count = 0
# loss_list = []
# thresh_hold_count = 0
# for epoch in range(EPOCHS) :
#     lstm.train()
#     for i, (inputs, labels) in enumerate(train_loader):
#         # Convert torch tensor to Variable
#         varIn = Variable(inputs).view(-1,TIME_STEP,L)
#         varTar = Variable(labels)
#         optimizer.zero_grad()                                   # Clear off the gradient in (w = w - gradient)
#         loss = loss_fun(lstm(varIn), varTar.float())            # Calculate the loss
#         loss.backward()                                         # Backpropagation
#         optimizer.step()                                        # Update the weights
#         if i % EPOCHS == 0 :
#             loss_list.append(loss)
#     lstm.eval()
#     for i, (inputs, labels) in enumerate(train_loader):
#         if i % EPOCHS == 0 :
#             train_pred_y = lstm(Variable(train_tensor_x).view(-1,TIME_STEP,L)).cpu().data.numpy()
#             # print(train_pred_y)
#             train_pred_y = np.array([0 if i<=0.5 else 1 for i in train_pred_y])
#             test_pred_y = lstm(Variable(test_tensor_x).view(-1,TIME_STEP,L)).cpu().data.numpy()
#             test_pred_y = np.array([0 if i<=0.5 else 1 for i in test_pred_y])
#             train_acc = np.sum(train_pred_y == train_tensor_y.cpu().data.numpy()) / train_pred_y.shape[0]
#             test_acc = np.sum(test_pred_y == test_tensor_y.cpu().data.numpy()) / train_pred_y.shape[0]
#             train_acc_arr = np.append(train_acc_arr,train_acc)
#             test_acc_arr = np.append(test_acc_arr,test_acc)
#             print("EPOCH : ", epoch, "training_acc : {:.2f}".format(train_acc),"testing_acc : {:.2f}".format(test_acc))
#     #     if(train_acc>=0.95):
#     #         thresh_hold_count+= 1
#     #     if thresh_hold_count>800:
#     #         break
#     # else:
#     #     continue
#     # break
#
# plt.figure();plt.plot(list(train_acc_arr));plt.title('LSTM train accuracy');plt.savefig('LSTM_train_accuracy_batch_{:f}_lr_{:f}_L_{:f}.png'.format(BATCH,LR,L))
# plt.figure();plt.plot(list(test_acc_arr));plt.title('LSTM test accuracy');plt.savefig('LSTM_test_accuracy_batch_{:f}_lr_{:f}_L_{:f}.png'.format(BATCH,LR,L))
# torch.save(lstm.state_dict(),'./LSTM_training_model_batch_{:f}_lr_{:f}_L_{:f}'.format(BATCH,LR,L))
#
#
#
# # ------------------------GRU------------------------------------
# print('GRU Learning Process')
#
# gru = GRU(in_dim=L).cuda()
# optimizer = torch.optim.Adam(gru.parameters(), lr = LR)
# loss_fun = nn.MSELoss()
#
# train_acc_arr = np.empty((0))
# test_acc_arr = np.empty((0))
#
# loss_list = []
# thresh_hold_count = 0
# for epoch in range(EPOCHS) :
#     gru.train()
#     for i, (inputs, labels) in enumerate(train_loader):
#         # Convert torch tensor to Variable
#         varIn = Variable(inputs).view(-1,TIME_STEP,L)
#         varTar = Variable(labels)
#         optimizer.zero_grad()                                   # Clear off the gradient in (w = w - gradient)
#         loss = loss_fun(gru(varIn), varTar.float())            # Calculate the loss
#         loss.backward()                                         # Backpropagation
#         optimizer.step()                                        # Update the weights
#         if i % EPOCHS == 0 :
#             loss_list.append(loss)
#
#     gru.eval()
#     for i, (inputs, labels) in enumerate(train_loader):
#         if i % EPOCHS == 0 :
#             train_pred_y = gru(Variable(train_tensor_x).view(-1,TIME_STEP,L)).cpu().data.numpy()
#             # print(train_pred_y)
#             train_pred_y = np.array([0 if i<=0.5 else 1 for i in train_pred_y])
#             test_pred_y = gru(Variable(test_tensor_x).view(-1,TIME_STEP,L)).cpu().data.numpy()
#             test_pred_y = np.array([0 if i<=0.5 else 1 for i in test_pred_y])
#             train_acc = np.sum(train_pred_y == train_tensor_y.cpu().data.numpy()) / train_pred_y.shape[0]
#             test_acc = np.sum(test_pred_y == test_tensor_y.cpu().data.numpy()) / train_pred_y.shape[0]
#             train_acc_arr = np.append(train_acc_arr,train_acc)
#             test_acc_arr = np.append(test_acc_arr,test_acc)
#             print("EPOCH : ", epoch, "training_acc : {:.2f}".format(train_acc),"testing_acc : {:.2f}".format(test_acc))
#     #     if(train_acc>=0.95):
#     #         thresh_hold_count+= 1
#     #     if thresh_hold_count>800:
#     #         break
#     # else:
#     #     continue
#     # break
#
# plt.figure();plt.plot(list(train_acc_arr));plt.title('GRU train accuracy');plt.savefig('GRU_train_accuracy_batch_{:f}_lr_{:f}_L_{:f}.png'.format(BATCH,LR,L))
# plt.figure();plt.plot(list(test_acc_arr));plt.title('GRU test accuracy');plt.savefig('GRU_test_accuracy_batch_{:f}_lr_{:f}_L_{:f}.png'.format(BATCH,LR,L))
# torch.save(gru.state_dict(),'./GRU_training_model_batch_{:f}_lr_{:f}_L_{:f}'.format(BATCH,LR,L))

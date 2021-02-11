image_size = 224
finetune_data_path = './fine_tune_list.txt'
train_data_path = './train_list.txt'



T_batch_size = 64
T_class_num = 17
T_weithts_path = './A1_train_model/T_model.h5'

F_batch_size = 256
F_class_num = 3
F_svm_threshold = 0.3
F_fineturn_threshold = 0.3
F_regression_threshold = 0.6
F_weithts_path = './A2_train_model/F_model.h5'

SVM_and_Reg_save_path = './flowerData/Reg'
fineturn_save_path = './flowerData/Finetune'
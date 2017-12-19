../../build/tools/convert_driving_data $PATH_TO_DATASET_DIR$ train_list.txt LMDB_train
../../build/tools/compute_driving_mean LMDB_train ./driving_mean_train.binaryproto lmdb

../../build/tools/convert_driving_data $PATH_TO_DATASET_DIR$ test_list.txt LMDB_test



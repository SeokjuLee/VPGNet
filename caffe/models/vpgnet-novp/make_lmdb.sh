../../build/tools/convert_driving_data /media/rcv/HDD/caltech-lanes /media/rcv/HDD/caltech-lanes/cordova1.txt LMDB_train
../../build/tools/compute_driving_mean LMDB_train ./driving_mean_train.binaryproto lmdb

../../build/tools/convert_driving_data /media/rcv/HDD/caltech-lanes /media/rcv/HDD/caltech-lanes/cordova2.txt LMDB_test



// This program converts detection bounding box labels to Dataum proto buffers
// and save them in LMDB.
// Usage:
//   convert_annotation_data [FLAGS] id_list_file annotation_list_file img_list_file type_list_file DB_NAME
//

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>
#include <cstdio>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#undef NDEBUG
#include <cassert>

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;
using google::protobuf::Message;

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    if (s.size()>0 && s[s.size()-1] == delim) elems.push_back("");
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

int find_type(std::vector<string> &types, string tp) {
    int id=-1;
    for (int i=0; i<types.size(); i++)
        if (tp == types[i]) {
            id = i;
            break;
        }
    if (id == -1) {
        LOG(INFO) << "new type " << tp << " " << types.size();
        id = types.size();
        types.push_back(tp);
    }
    return id;
}

DEFINE_bool(test_run, false, "If set to true, only generate 100 images.");
DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, true,
    "Randomly shuffle the order of images and their labels");
DEFINE_bool(use_rgb, false, "use RGB channels");
DEFINE_int32(resize_width, 640 + 32, "Width images are resized to");
DEFINE_int32(resize_height, 480 + 32, "Height images are resized to");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_annotation_data [FLAGS] id_list_file annotation_list_file img_list_file type_list_file DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 6) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_annotation_data");
    return 1;
  }

  bool is_color = !FLAGS_gray;
// load img list
  std::ifstream imglist_file(argv[3]);
  string filename;
  std::map<string,string> id2file;
  while (imglist_file >> filename) {
      int a = filename.find_last_of('/');
      string basename = filename.substr(a+1);
      size_t b = basename.find_last_of('.');
      string id = basename.substr(0, b);
      id2file[id] = filename;
      //LOG(INFO) << id << " -> " << filename << " " << basename << ' ' << a << ' ' << b;
  }
  imglist_file.close();
// load types
  std::vector<string> types;
  std::ifstream types_file(argv[4]);
  while (types_file >> filename)
      types.push_back(filename);
  types_file.close();
// load annotation file
  std::ifstream anno_file(argv[2]);
  string tmp;
  std::vector<string> tmps;
  std::map< string, caffe::DrivingData > id2data;
  std::getline(anno_file, tmp);
  string newfile = "########## NEW FILE ##########";
  assert(tmp==newfile);
  bool eof=false;
  while (!eof) {
      std::getline(anno_file, tmp, ' ');
      assert(tmp == "file:");
      std::getline(anno_file, tmp);
      size_t pos = tmp.find_last_of('/');
      if (pos == string::npos) {
          pos = tmp.find_last_of('\\');
      }
      assert(pos != string::npos);
      size_t pos2 = tmp.find_last_of('.');
      assert(pos2 != string::npos);
      string id = tmp.substr(pos+1, pos2-pos-1);

      assert(std::getline(anno_file, tmp));
      assert(tmp == "");

      caffe::DrivingData data;
      data.Clear();

      while (1) {
          caffe::CarBoundingBox box;
          box.Clear();
          eof = !std::getline(anno_file, tmp);
          if (eof) break;
          if (tmp == newfile)
              break;
          assert(tmp.substr(0,6) == "object");
          std::getline(anno_file, tmp);
          assert(tmp.substr(0,4) == "bbox");
          tmps = split(tmp.substr(6), ',');
          assert(tmps.size() == 4);
          box.set_xmin(std::atof(tmps[0].c_str()));
          box.set_ymin(std::atof(tmps[1].c_str()));
          box.set_xmax(box.xmin()+std::atof(tmps[2].c_str()));
          box.set_ymax(box.ymin()+std::atof(tmps[3].c_str()));
          std::getline(anno_file, tmp);
          assert(tmp.substr(0,8) == "category");
          tmps = split(tmp, ' ');
          box.set_type(find_type(types, tmps[1]));
          while (1) {
              std::getline(anno_file, tmp);
              if (tmp == "") break;
              tmps = split(tmp, ' ');
              assert(tmps.size() == 2);
              std::vector<string> pts = split(tmps[1], ',');
              std::vector<float> pt;
              for (int i=0; i<pts.size(); i++)
                  pt.push_back(std::atof(pts[i].c_str()));
              assert(pt.size()%2==0);
              caffe::FixedPoint *fp;
              if (tmps[0] == "ellipse:") {
                  assert(box.ellipse_mask_size() == 0);
                  for (int i=0; i<pt.size(); i+=2) {
                      fp = box.add_ellipse_mask();
                      fp->set_x(pt[i]);
                      fp->set_y(pt[i+1]);
                  }
              } else
              if (tmps[0] == "polygon:") {
                  assert(box.poly_mask_size() == 0);
                  for (int i=0; i<pt.size(); i+=2) {
                      fp = box.add_poly_mask();
                      fp->set_x(pt[i]);
                      fp->set_y(pt[i+1]);
                  }
              } else
              if (tmps[0] == "fixpoints:") {
                  LOG(INFO) << "warning \"" << tmp << '"';
              } else
                  assert(0);
          }
          assert(box.has_type());
          assert(box.has_xmax());
          if (box.ellipse_mask_size()==0 && box.poly_mask_size()==0) {
              LOG(ERROR) << "warning: no mask " << id;
          }
          //assert(box.ellipse_mask_size()>0 || box.poly_mask_size()>0);
          data.add_car_boxes()->CopyFrom(box);
      }
      id2data[id] = data;
  }

  std::ifstream idfile(argv[1]);
  std::vector<string> ids;
  while (idfile >> filename) {
      ids.push_back(filename);
  }
  idfile.close();

  LOG(INFO) << "save types to " << argv[4];
  std::ofstream types_file2(argv[4]);
  for (int i=0; i<types.size(); i++)
      types_file2 << types[i] << '\n';
  types_file2.close();

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(ids.begin(), ids.end());
  }
  LOG(INFO) << "A total of " << ids.size() << " images.";

  const char* db_path = argv[5];

  bool generate_img = true;
  std::string db_str(db_path);
  if (db_str == "none") {
    generate_img = false;
  }

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Open new db
  // lmdb
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;

  // Open db
  LOG(INFO) << "Opening lmdb " << db_path;
  CHECK_EQ(mkdir(db_path, 0744), 0)
      << "mkdir " << db_path << "failed";
  CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
      << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
      << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
      << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
      << "mdb_open failed. Does the lmdb already exist? ";

  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int count = 0;
  LOG(ERROR) << "Total to be processed: " << ids.size() << ".\n";

  for (int i = 0; i < ids.size(); i++) {
    DrivingData data;
    string id = ids[i];
    assert(id2file.count(id));
    assert(id2data.count(id));
    data.CopyFrom(id2data[id]);
    if (data.car_boxes_size()==0) {
        LOG(ERROR) << "error, no box inside " << id;
        continue;
    }
  }
  for (int i = 0; i < ids.size(); i++) {
    DrivingData data;
    string id = ids[i];
    assert(id2file.count(id));
    assert(id2data.count(id));
    data.CopyFrom(id2data[id]);
    if (data.car_boxes_size()==0) {
        continue;
    }
    const string image_path = id2file[id];
    data.set_car_img_source(image_path);

    if (!ReadImageToDatum(image_path, 0,
        resize_height, resize_width, is_color, data.mutable_car_image_datum())) {
      LOG(INFO) << "read failed " << image_path;
      continue;
    }

    // sequential
    snprintf(key_cstr, kMaxKeyLength, "%08d_%s", i,
        id.c_str());
    string value;
    data.SerializeToString(&value);
    string keystr(key_cstr);

    // Put in db
    mdb_data.mv_size = value.size();
    mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
    mdb_key.mv_size = keystr.size();
    mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
    CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
        << "mdb_put failed";

    if (++count % 1000 == 0) {
      // Commit txn
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
          << "mdb_txn_commit failed";
      CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
          << "mdb_txn_begin failed";
      LOG(ERROR) << "Processed " << count << " files.";
    } else
    if (count % 10 == 0) {
        LOG(ERROR) << "Processed " << count << " files.";
    }

    if (FLAGS_test_run && count == 10) {
      break;
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
    mdb_close(mdb_env, mdb_dbi);
    mdb_env_close(mdb_env);
    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}

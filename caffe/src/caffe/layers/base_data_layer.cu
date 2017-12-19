#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data());
  // Copy the data
  caffe_copy(batch->data().count(), batch->data().gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    for (int i=1; i<top.size(); i++) {
        // Reshape to loaded labels.
        top[i]->ReshapeLike(batch->label(i-1));
        // Copy the labels.
        caffe_copy(batch->label(i-1).count(), batch->label(i-1).gpu_data(),
            top[i]->mutable_gpu_data());
    }
  }

  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe

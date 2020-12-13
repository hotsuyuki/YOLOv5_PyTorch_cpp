// This code follows Google C++ Style Guide

#include "object_detector.h"

#include <dirent.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>


namespace yolov5 {

bool ObjectDetector::LoadClassNames(const std::string& class_name_filename) {
  std::ifstream class_name_ifs(class_name_filename);
  if (class_name_ifs.is_open()) {
    std::string class_name;
    while (std::getline(class_name_ifs, class_name)) {
      class_names_.emplace_back(class_name);
    }
    class_name_ifs.close();
  } else {
    std::cerr << "[ObjectDetector::LoadClassNames()] Error: Could not open "
              << class_name_filename << "\n";
    return false;
  }
  
  if (class_names_.size() == 0) {
    std::cerr << "[ObjectDetector::LoadClassNames()] Error: labe names are empty \n";
    return false;
  }

  return true;
}


bool ObjectDetector::LoadInputImagePaths(const std::string& input_directory) {
  DIR* dir;
  struct dirent* entry;
  if ((dir = opendir(input_directory.c_str())) != NULL) {
    while ((entry = readdir(dir)) != NULL) {
      if (entry->d_name[0] != '.') {
        std::string input_image_filename(entry->d_name);
        std::string input_image_path = input_directory + input_image_filename;
        input_image_paths_.emplace_back(input_image_path);
      }
    }
    closedir(dir);
  } else {
    std::cerr << "[ObjectDetector::LoadInputImages()] Error: Could not open "
              << input_directory << "\n";
    return false;
  }
 
  if (input_image_paths_.size() == 0) {
    std::cerr << "[ObjectDetector::LoadInputImages()] Error: input image filenames are empty \n";
    return false;
  }

  return true;
}


void ObjectDetector::Inference(float confidence_threshold, float iou_threshold) {
  std::cout << "=== Empty inferences to warm up === \n\n";
  for (std::size_t i = 0; i < 3; ++i) {
    cv::Mat tmp_image = cv::Mat::zeros(input_height_, input_width_, CV_32FC3);
    std::vector<ObjectInfo> tmp_results;
    Detect(tmp_image, 1.0, 1.0, tmp_results);
  }
  std::cout << "=== Warming up is done === \n\n\n";

  for (const auto& input_image_path : input_image_paths_) {
    std::cout << "input_image_path = " << input_image_path << "\n";

    cv::Mat input_image = cv::imread(input_image_path);
    if (input_image.empty()) {
      std::cerr << "[ObjectDetector::Run()] Error: Cloud not open "
                << input_image_path << "\n";
      continue;
    }

    std::vector<ObjectInfo> results;
    Detect(input_image, confidence_threshold, iou_threshold, results);

    SaveResultImage(input_image, results, input_image_path);
  }

  return;
}


void ObjectDetector::Detect(const cv::Mat& input_image,
                            float confidence_threshold, float iou_threshold,
                            std::vector<ObjectInfo>& results) {
  torch::NoGradGuard no_grad_guard;

  auto start_preprocess = std::chrono::high_resolution_clock::now();
  std::vector<torch::jit::IValue> inputs;
  LetterboxInfo letterbox_info = PreProcess(input_image, inputs);
  auto end_preprocess = std::chrono::high_resolution_clock::now();

  auto duration_preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(end_preprocess - start_preprocess);
  std::cout << "Pre-processing: " << duration_preprocess.count() << " [ms] \n";

  // output_tensor ... {Batch=1, Num of max bbox=25200, 85}
  // 25200 ... {(640[px]/32[stride])^2 + (640[px]/16[stride])^2 + (640[px]/8[stride])^2} x 3[layer]
  // 85 ... 0: center x, 1: center y, 2: width, 3: height, 4: obj conf, 5~84: class conf 
  auto start_inference = std::chrono::high_resolution_clock::now();
  at::Tensor output_tensor = model_.forward(inputs).toTuple()->elements()[0].toTensor();
  auto end_inference = std::chrono::high_resolution_clock::now();

  auto duration_inference = std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference);
  std::cout << "Inference: " << duration_inference.count() << " [ms] \n";

  // results ... {Num of obj, 6}
  // 6 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y, 4: class score, 5: class id
  auto start_postprocess = std::chrono::high_resolution_clock::now();
  PostProcess(output_tensor, letterbox_info, confidence_threshold, iou_threshold, results);
  auto end_postprocess = std::chrono::high_resolution_clock::now();

  auto duration_postprocess = std::chrono::duration_cast<std::chrono::milliseconds>(end_postprocess - start_postprocess);
  std::cout << "Post-processing: " << duration_postprocess.count() << " [ms] \n\n";

  return;
}


LetterboxInfo ObjectDetector::PreProcess(const cv::Mat& input_image,
                                         std::vector<torch::jit::IValue>& inputs) {
  cv::Mat letterbox_image;
  LetterboxInfo letterbox_info = Letterboxing(input_image, letterbox_image);

  // 0 ~ 255 ---> 0.0 ~ 1.0
  cv::cvtColor(letterbox_image, letterbox_image, cv::COLOR_BGR2RGB);
  letterbox_image.convertTo(letterbox_image, CV_32FC3, 1.0 / 255.0);

  // input_tensor ... {Batch=1, Height, Width, Channel=3}
  // --->
  // input_tensor ... {Batch=1, Channel=3, Height, Width}
  at::Tensor input_tensor = torch::from_blob(letterbox_image.data,
                                             {1, input_height_, input_width_, 3});
  input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();

  if (is_gpu_) {
    input_tensor = input_tensor.to(torch::kCUDA);
    input_tensor = input_tensor.to(torch::kHalf);
  } else {
    input_tensor = input_tensor.to(torch::kCPU);
  }

  inputs.clear();
  inputs.emplace_back(input_tensor);

  return letterbox_info;
}


LetterboxInfo ObjectDetector::Letterboxing(const cv::Mat& input_image, cv::Mat& letterbox_image) {
  float scale = std::min(input_height_ / static_cast<float>(input_image.size().height),
                         input_width_ / static_cast<float>(input_image.size().width));
  cv::resize(input_image, letterbox_image, cv::Size(), scale, scale);

  int top_margin = floor((input_height_ - letterbox_image.size().height) / 2.0);
  int bottom_margin = ceil((input_height_ - letterbox_image.size().height) / 2.0);
  int left_margin = floor((input_width_ - letterbox_image.size().width) / 2.0);
  int right_margin = ceil((input_width_ - letterbox_image.size().width) / 2.0);
  cv::copyMakeBorder(letterbox_image, letterbox_image,
                     top_margin, bottom_margin, left_margin, right_margin,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

  LetterboxInfo letterbox_info;
  letterbox_info.original_height = input_image.size().height;
  letterbox_info.original_width = input_image.size().width;
  letterbox_info.scale = scale;
  letterbox_info.padding_height = top_margin;
  letterbox_info.padding_width = left_margin;

  return letterbox_info;
}


void ObjectDetector::PostProcess(const at::Tensor& output_tensor, const LetterboxInfo& letterbox_info,
                                 float confidence_threshold, float iou_threshold,
                                 std::vector<ObjectInfo>& results) {
  int batch_size = output_tensor.size(0);
  if (batch_size != 1) {
    std::cerr << "[ObjectDetector::PostProcess()] Error: Batch size of output tensor is not 1 \n";
    return;
  }

  // 85 ... 0: center x, 1: center y, 2: width, 3: height, 4: obj conf, 5~84: class conf
  int num_bbox_confidence_class_idx = output_tensor.size(2);

  // 5 = 85 - 80 ... 0: center x, 1: center y, 2: width, 3: height, 4: obj conf
  int num_bbox_confidence_idx = num_bbox_confidence_class_idx - class_names_.size();

  // 4 = 5 - 1 ... 0: center x, 1: center y, 2: width, 3: height
  int num_bbox_idx = num_bbox_confidence_idx - 1;



  /*****************************************************************************
   * Thresholding the detected objects by class confidence
   ****************************************************************************/

  int bbox_confidence_class_dim = -1;  // always in the last dimension
  int object_confidence_idx = 4;

  // output_tensor ... {Batch=1, Num of max bbox=25200, 85}
  // --->
  // candidate_object_mask ... {Batch=1, Num of max bbox=25200, 1}
  at::Tensor candidate_object_mask = output_tensor.select(bbox_confidence_class_dim,
                                                          object_confidence_idx);
  candidate_object_mask = candidate_object_mask.gt(confidence_threshold);
  candidate_object_mask = candidate_object_mask.unsqueeze(bbox_confidence_class_dim);

  // output_tensor[0] ... {Num of max bbox=25200, 85}
  // candidate_object_mask[0] ... {Num of max bbox=25200, 1}
  // --->
  // candidate_object_tensor ... {Num of candidate bbox*85}
  at::Tensor candidate_object_tensor = torch::masked_select(output_tensor[0],
                                                            candidate_object_mask[0]);

  // candidate_object_tensor ... {Num of candidate bbox*85}
  // --->
  // candidate_object_tensor ... {Num of candidate bbox, 85}
  candidate_object_tensor = candidate_object_tensor.view({-1, num_bbox_confidence_class_idx});

  // If there is no any candidate objects at all, return
  if (candidate_object_tensor.size(0) == 0) {
    return;
  }

  // candidate_object_tensor ... {Num of candidate bbox, 85}
  // --->
  // xywh_bbox_tensor ... {Num of candidate bbox, 4} => similar to [:, 0:4] in Python
  at::Tensor xywh_bbox_tensor = candidate_object_tensor.slice(bbox_confidence_class_dim,
                                                              0, num_bbox_idx);

  // xywh_bbox_tensor ... {Num of candidate bbox, 4}
  // 4 ... 0: x center, 1: y center, 2: width, 3: height
  // --->
  // bbox_tensor ... {Num of candidate bbox, 4}
  // 4 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y
  at::Tensor bbox_tensor;
  XcenterYcenterWidthHeight2TopLeftBottomRight(xywh_bbox_tensor, bbox_tensor);

  // candidate_object_tensor ... {Num of candidate bbox, 85}
  // --->
  // object_confidence_tensor ... {Num of candidate bbox, 1} => similar to [:, 4:5] in Python
  at::Tensor object_confidence_tensor = candidate_object_tensor.slice(bbox_confidence_class_dim,
                                                                      num_bbox_idx, num_bbox_confidence_idx);

  // candidate_object_tensor ... {Num of candidate bbox, 85}
  // --->
  // class_confidence_tensor ... {Num of candidate bbox, 80} => similar to [:, 5:] in Python
  at::Tensor class_confidence_tensor = candidate_object_tensor.slice(bbox_confidence_class_dim,
                                                                     num_bbox_confidence_idx);

  // class_score_tensor ... {Num of candidate bbox, 80}
  at::Tensor class_score_tensor = class_confidence_tensor * object_confidence_tensor;

  // max_class_score_tuple ... (value: {Num of candidate bbox}, index: {Num of candidate bbox})
  std::tuple<at::Tensor, at::Tensor> max_class_score_tuple = torch::max(class_score_tensor,
                                                                        bbox_confidence_class_dim);

  // max_class_score ... {Num of candidate bbox}
  // ---> 
  // max_class_score ... {Num of candidate bbox, 1}
  at::Tensor max_class_score = std::get<0>(max_class_score_tuple).to(torch::kFloat);
  max_class_score = max_class_score.unsqueeze(bbox_confidence_class_dim);

  // max_class_id ... {Num of candidate bbox}
  // --->
  // max_class_id ... {Num of candidate bbox, 1}
  at::Tensor max_class_id = std::get<1>(max_class_score_tuple).to(torch::kFloat);
  max_class_id = max_class_id.unsqueeze(bbox_confidence_class_dim);

  // result_tensor ... {Num of candidate bbox, 6}
  // 6 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y, 4: class score, 5: class id
  at::Tensor result_tensor = torch::cat({bbox_tensor, max_class_score, max_class_id},
                                        bbox_confidence_class_dim);



  /*****************************************************************************
   * Non Maximum Suppression
   ****************************************************************************/

  // class_id_tensor ... {Num of candidate bbox, 1} => similar to [:, -1:] in Python
  at::Tensor class_id_tensor = result_tensor.slice(bbox_confidence_class_dim, -1);

  // class_offset_bbox_tensor ... {Num of candidate bbox, 4}
  // 4 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y (but offset by +4096 * class id)
  at::Tensor class_offset_bbox_tensor = result_tensor.slice(bbox_confidence_class_dim, 0, num_bbox_idx)
                                          + nms_max_bbox_size_ * class_id_tensor;
  
  // Copies tensor to CPU to access tensor elements efficiently with TensorAccessor
  // https://pytorch.org/cppdocs/notes/tensor_basics.html#efficient-access-to-tensor-elements
  at::Tensor class_offset_bbox_tensor_cpu = class_offset_bbox_tensor.cpu();
  at::Tensor result_tensor_cpu = result_tensor.cpu();
  auto class_offset_bbox_tensor_accessor = class_offset_bbox_tensor_cpu.accessor<float, 2>();
  auto result_tensor_accessor = result_tensor_cpu.accessor<float, 2>();

  std::vector<cv::Rect> offset_bboxes;
  std::vector<float> class_scores;
  offset_bboxes.reserve(result_tensor_accessor.size(0));
  class_scores.reserve(result_tensor_accessor.size(0));

  for (std::size_t i = 0; i < result_tensor_accessor.size(0); ++i) {
    float class_offset_top_left_x = class_offset_bbox_tensor_accessor[i][0];
    float class_offset_top_left_y = class_offset_bbox_tensor_accessor[i][1];
    float class_offset_bottom_right_x = class_offset_bbox_tensor_accessor[i][2];
    float class_offset_bottom_right_y = class_offset_bbox_tensor_accessor[i][3];

    offset_bboxes.emplace_back(cv::Rect(cv::Point(class_offset_top_left_x, class_offset_top_left_y),
                                        cv::Point(class_offset_bottom_right_x, class_offset_bottom_right_y)));

    class_scores.emplace_back(result_tensor_accessor[i][4]);
  }

  std::vector<int> nms_indecies;
  cv::dnn::NMSBoxes(offset_bboxes, class_scores, confidence_threshold, iou_threshold, nms_indecies);



  /*****************************************************************************
   * Create result data
   ****************************************************************************/

  std::vector<ObjectInfo> object_infos;
  for (const auto& nms_idx : nms_indecies) {
    float top_left_x = result_tensor_accessor[nms_idx][0];
    float top_left_y = result_tensor_accessor[nms_idx][1];
    float bottom_right_x = result_tensor_accessor[nms_idx][2];
    float bottom_right_y = result_tensor_accessor[nms_idx][3];

    ObjectInfo object_info;
    object_info.bbox_rect = cv::Rect(cv::Point(top_left_x, top_left_y),
                                      cv::Point(bottom_right_x, bottom_right_y));
    object_info.class_score = result_tensor_accessor[nms_idx][4];
    object_info.class_id = result_tensor_accessor[nms_idx][5];

    object_infos.emplace_back(object_info);
  }

  RestoreBoundingboxSize(object_infos, letterbox_info, results);

  return;
}


// xywh_bbox_tensor ... {Num of bbox, 4}
// 4 ... 0: x center, 1: y center, 2: width, 3: height
// --->
// tlbr_bbox_tensor ... {Num of bbox, 4}
// 4 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y
void ObjectDetector::XcenterYcenterWidthHeight2TopLeftBottomRight(const at::Tensor& xywh_bbox_tensor,
                                                                  at::Tensor& tlbr_bbox_tensor) {
  tlbr_bbox_tensor = torch::zeros_like(xywh_bbox_tensor);

  int bbox_dim = -1;  // the last dimension

  int x_center_idx = 0;
  int y_center_idx = 1;
  int width_idx = 2;
  int height_idx = 3;

  tlbr_bbox_tensor.select(bbox_dim, 0) = xywh_bbox_tensor.select(bbox_dim, x_center_idx)
                                           - xywh_bbox_tensor.select(bbox_dim, width_idx).div(2.0);
  tlbr_bbox_tensor.select(bbox_dim, 1) = xywh_bbox_tensor.select(bbox_dim, y_center_idx)
                                           - xywh_bbox_tensor.select(bbox_dim, height_idx).div(2.0);
  tlbr_bbox_tensor.select(bbox_dim, 2) = xywh_bbox_tensor.select(bbox_dim, x_center_idx)
                                           + xywh_bbox_tensor.select(bbox_dim, width_idx).div(2.0);
  tlbr_bbox_tensor.select(bbox_dim, 3) = xywh_bbox_tensor.select(bbox_dim, y_center_idx)
                                           + xywh_bbox_tensor.select(bbox_dim, height_idx).div(2.0);
  
  return;
}


void ObjectDetector::RestoreBoundingboxSize(const std::vector<ObjectInfo>& object_infos,
                                            const LetterboxInfo& letterbox_info,
                                            std::vector<ObjectInfo>& restored_object_infos) {
  restored_object_infos.clear();
  restored_object_infos.reserve(object_infos.size());

  for (const auto& object_info : object_infos) {
    float top_left_x = (object_info.bbox_rect.tl().x - letterbox_info.padding_width) / letterbox_info.scale;
    float top_left_y = (object_info.bbox_rect.tl().y - letterbox_info.padding_height) / letterbox_info.scale;
    float bottom_right_x = (object_info.bbox_rect.br().x - letterbox_info.padding_width) / letterbox_info.scale;
    float bottom_right_y = (object_info.bbox_rect.br().y - letterbox_info.padding_height) / letterbox_info.scale;

    top_left_x = CLAMP(0.0f, top_left_x, static_cast<float>(letterbox_info.original_width));
    top_left_y = CLAMP(0.0f, top_left_y, static_cast<float>(letterbox_info.original_height));
    bottom_right_x = CLAMP(0.0f, bottom_right_x, static_cast<float>(letterbox_info.original_width));
    bottom_right_y = CLAMP(0.0f, bottom_right_y, static_cast<float>(letterbox_info.original_height));

    ObjectInfo restored_object_info;
    restored_object_info.bbox_rect = cv::Rect(cv::Point(top_left_x, top_left_y),
                                              cv::Point(bottom_right_x, bottom_right_y));
    restored_object_info.class_score = object_info.class_score;
    restored_object_info.class_id = object_info.class_id;

    restored_object_infos.emplace_back(restored_object_info);
  }

  return;
}


void ObjectDetector::SaveResultImage(const cv::Mat& input_image,
                                     const std::vector<ObjectInfo>& results,
                                     const std::string& input_image_path) {
  cv::Mat result_image(input_image);

  for (const auto& object_info : results) {
    // Draws object bounding box
    cv::rectangle(result_image, object_info.bbox_rect, cv::Scalar(0,0,255), 1);

    // Class info text
    std::string class_name = class_names_[object_info.class_id];
    std::stringstream class_score;
    class_score << std::fixed << std::setprecision(2) << object_info.class_score;
    std::string class_info = class_name + " " + class_score.str();

    // Size of class info text
    auto font_face = cv::FONT_HERSHEY_SIMPLEX;
    float font_scale = 1.0;
    int thickness = 1;
    int baseline = 0;
    cv::Size class_info_size = cv::getTextSize(class_info, font_face, font_scale, thickness, &baseline);

    // Draws rectangle of class info text
    int height_offset = 5;  // [px]
    cv::Point class_info_top_left = cv::Point(object_info.bbox_rect.tl().x,
                                              object_info.bbox_rect.tl().y - class_info_size.height - height_offset);
    cv::Point class_info_bottom_right = cv::Point(object_info.bbox_rect.tl().x + class_info_size.width,
                                                  object_info.bbox_rect.tl().y);
    cv::rectangle(result_image, class_info_top_left, class_info_bottom_right, cv::Scalar(0,0,255), -1);

    // Draws class info text
    cv::Point class_info_text_position = cv::Point(object_info.bbox_rect.tl().x,
                                                   object_info.bbox_rect.tl().y - height_offset);
    cv::putText(result_image, class_info, class_info_text_position,
                font_face, font_scale, cv::Scalar(0,0,0), thickness);
  }

  std::size_t last_slash_pos = input_image_path.find_last_of('/');
  std::string input_image_directory = input_image_path.substr(0, last_slash_pos + 1);
  std::string input_image_filename = input_image_path.substr(last_slash_pos + 1);

  std::size_t last_hyphen_pos = input_image_filename.find_last_of('_');
  std::string result_image_directory = input_image_directory + "../result_image/";
  std::string result_image_filename = "result_" + input_image_filename.substr(last_hyphen_pos + 1);

  cv::imwrite(result_image_directory + result_image_filename, result_image);

  return;
}

}  // namespace yolov5
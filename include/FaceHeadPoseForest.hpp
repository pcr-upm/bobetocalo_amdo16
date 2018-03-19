/** ****************************************************************************
 *  @file    FaceHeadPoseForest.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_HEADPOSE_FOREST_HPP
#define FACE_HEADPOSE_FOREST_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceHeadPose.hpp>
#include <HeadPoseSample.hpp>
#include <Forest.hpp>
#include <ImageSample.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

/** ****************************************************************************
 * @class FaceHeadPoseForest
 * @brief Class used for head pose estimation.
 ******************************************************************************/
class FaceHeadPoseForest: public FaceHeadPose
{
public:
  FaceHeadPoseForest(std::string path);

  ~FaceHeadPoseForest() {};

  void
  parseOptions
    (
    int argc,
    char **argv
    );

  void
  train
    (
    const std::vector<upm::FaceAnnotation> &anns_train,
    const std::vector<upm::FaceAnnotation> &anns_valid
    );

  void
  load();

  void
  process
    (
    cv::Mat frame,
    std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    );

private:
  cv::Point3f
  getHeadPoseVotesMT
    (
    boost::shared_ptr<ImageSample> sample,
    const Forest<HeadPoseSample> &forest
    );

  std::vector<int> hp_trees_idx;
  ForestParam hp_forest_param;
  Forest<HeadPoseSample> hp_forest;
};

} // namespace upm

#endif /* FACE_HEADPOSE_FOREST_HPP */

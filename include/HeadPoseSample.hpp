/** ****************************************************************************
 *  @file    HeadPoseSample.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef HEAD_POSE_SAMPLE_HPP
#define HEAD_POSE_SAMPLE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <ImageSample.hpp>
#include <SplitGen.hpp>
#include <ThresholdSplit.hpp>
#include <vector>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <opencv2/highgui/highgui.hpp>

class HeadPoseLeaf;

/** ****************************************************************************
 * @class HeadPoseSample
 * @brief Head pose sample
 ******************************************************************************/
class HeadPoseSample
{
public:
  typedef ThresholdSplit<SimplePatchFeature> Split;
  typedef HeadPoseLeaf Leaf;

  /**
   * @brief Training patch constructor. Save the angle head-pose of interest.
   */
  HeadPoseSample
    (
    boost::shared_ptr<ImageSample> sample,
    cv::Rect patch_bbox,
    cv::Point3f label
    ) :
      m_image(sample), m_patch_bbox(patch_bbox), m_label(label) {};

  /**
   * @brief Testing patch constructor for dense extraction.
   */
  HeadPoseSample
    (
    boost::shared_ptr<ImageSample> sample,
    cv::Rect patch_bbox
    ) :
      m_image(sample), m_patch_bbox(patch_bbox), m_label(0,0,0) {};

  /**
   * @brief Default head-pose patch destroyer.
   */
  virtual
  ~HeadPoseSample() {};

  /**
   * @brief Optional viewer useful to show training random patches.
   */
  void
  show();

  /**
   * @brief Simple patch feature comparison used to compute f_θ and split
   * samples in two different children.
   */
  int
  evalTest
    (
    const Split &test
    ) const;

  /**
   * @brief Check if f_θ is smaller than a threshold to generate left child.
   */
  bool
  eval
    (
    const Split &test
    ) const;

  /**
   * @brief Generate test split θ = (R1,R2,α) describe two rectangles within
   * the patch boundaries and the selected appearance channel.
   */
  static void
  generateSplit
    (
    const std::vector<HeadPoseSample*> &samples,
    boost::mt19937 *rng,
    cv::Size patch_size,
    Split &split
    );

  /**
   * @brief Select the splitting candidate which maximizes the evaluation
   * function information gain.
   */
  static double
  evalSplit
    (
    const std::vector<HeadPoseSample*> &setA,
    const std::vector<HeadPoseSample*> &setB
    );

  /**
   * @brief Create tree leaf node for head-pose estimation. Initialize
   * leaf node and compute histogram using a set of patches.
   */
  static void
  makeLeaf
    (
    HeadPoseLeaf &leaf,
    const std::vector<HeadPoseSample*> &set
    );

private:
  /**
   * @brief Entropy related to the head-pose angle within a set of patches.
   * The goal is to minimize the entropy between a set of patches with the
   * same orientation.
   */
  static double
  entropyHeadpose
    (
    const std::vector<HeadPoseSample*> &set
    );

  boost::shared_ptr<ImageSample> m_image;
  cv::Rect m_patch_bbox;
  cv::Point3f m_label;
};

/** ****************************************************************************
 * @class HeadPoseLeaf
 * @brief Head pose leaf sample
 ******************************************************************************/
class HeadPoseLeaf
{
public:
  HeadPoseLeaf() {};

  std::vector<int> hp_histogram;
  float hp_mean;
  float hp_variance;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & hp_histogram;
    ar & hp_mean;
    ar & hp_variance;
  }
};

#endif /* HEAD_POSE_SAMPLE_HPP */

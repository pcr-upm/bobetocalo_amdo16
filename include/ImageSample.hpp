/** ****************************************************************************
 *  @file    ImageSample.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef IMAGE_SAMPLE_HPP
#define IMAGE_SAMPLE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FeatureChannelFactory.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/serialization/access.hpp>
#include <opencv2/opencv.hpp>

namespace boost {
namespace serialization {

  template<class Archive, class T>
  void serialize(Archive &ar, cv::Rect_<T> &rect, const unsigned int version)
  {
    ar & rect.x;
    ar & rect.y;
    ar & rect.width;
    ar & rect.height;
  }

} // namespace serialization
} // namespace boost

struct SimplePatchFeature
{
  void
  generate
    (
    boost::mt19937 *rng,
    cv::Size patch_size,
    int num_channels = 0,
    float max_subpatch_ratio = 1.0f
    )
  {
    // Selected appearance channel randomly
    feature_channel = 0;
    if (num_channels > 1)
    {
      boost::uniform_int<> dist_feat(0, num_channels - 1);
      boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_feat(*rng, dist_feat);
      feature_channel = rand_feat();
    }

    // R1 and R2 describe two rectangles within the patch boundaries
    int subpatch_size = static_cast<int>(patch_size.height * max_subpatch_ratio);
    boost::uniform_int<> dist_size(1, (subpatch_size-1) * 0.75);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_size(*rng, dist_size);
    rect1.width  = rand_size();
    rect1.height = rand_size();
    rect2.width  = rand_size();
    rect2.height = rand_size();

    boost::uniform_int<> dist_x_a(0, subpatch_size-rect1.width-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_x_a(*rng, dist_x_a);
    rect1.x = rand_x_a();
    boost::uniform_int<> dist_y_a(0, subpatch_size-rect1.height-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_y_a(*rng, dist_y_a);
    rect1.y = rand_y_a();
    boost::uniform_int<> dist_x_b(0, subpatch_size-rect2.width-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_x_b(*rng, dist_x_b);
    rect2.x = rand_x_b();
    boost::uniform_int<> dist_y_b(0, subpatch_size-rect2.height-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_y_b(*rng, dist_y_b);
    rect2.y = rand_y_b();

    /*CV_Assert(rect1.x >= 0 && rect2.x >= 0 && rect1.y >= 0 && rect2.y >= 0);
    CV_Assert(rect1.x+rect1.width < patch_size.width && rect1.y+rect1.height < patch_size.height);
    CV_Assert(rect2.x+rect2.width < patch_size.width && rect2.y+rect2.height < patch_size.height);*/
  };

  int feature_channel;
  cv::Rect_<int> rect1;
  cv::Rect_<int> rect2;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & feature_channel;
    ar & rect1;
    ar & rect2;
  }
};

/** ****************************************************************************
 * @class ImageSample
 * @brief Patch sample from an image
 ******************************************************************************/
class ImageSample
{
public:
  ImageSample
    (
    const cv::Mat &img,
    std::vector<int> features,
    bool use_integral = false
    );

  virtual
  ~ImageSample();

  // Patch feature comparison (R1,R2,α) returns f_θ
  int
  evalTest
    (
    const SimplePatchFeature &test,
    const cv::Rect rect
    ) const;

  std::vector<cv::Mat> m_feature_channels;

private:
  bool m_use_integral;
};

#endif /* IMAGE_SAMPLE_HPP */

/** ****************************************************************************
 *  @file    HeadPoseSample.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <utils.hpp>
#include <HeadPoseSample.hpp>
#include <Constants.hpp>

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
HeadPoseSample::show()
{
  cv::Scalar white_color = cv::Scalar(255, 255, 255);
  cv::Mat img = m_image->m_feature_channels[0].clone();
  cv::imshow("Gray patch", img(m_patch_bbox));
  cv::rectangle(img, m_patch_bbox, white_color);
  cv::imshow("Face", img);
  cv::waitKey(0);
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
int
HeadPoseSample::evalTest
  (
  const Split &test
  ) const
{
  return m_image->evalTest(test.feature, m_patch_bbox);
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
bool
HeadPoseSample::eval
  (
  const Split &test
  ) const
{
  return evalTest(test) <= test.threshold;
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
HeadPoseSample::generateSplit
  (
  const std::vector<std::shared_ptr<HeadPoseSample>> &samples,
  boost::mt19937 *rng,
  cv::Size patch_size,
  Split &split
  )
{
  int num_channels = samples[0]->m_image->m_feature_channels.size();
  split.feature.generate(rng, patch_size, num_channels);
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
double
HeadPoseSample::evalSplit
  (
  const std::vector<std::shared_ptr<HeadPoseSample>> &setA,
  const std::vector<std::shared_ptr<HeadPoseSample>> &setB
  )
{
  double sizeA = setA.size(), sizeB = setB.size();
  double entropyA = entropyHeadpose(setA), entropyB = entropyHeadpose(setB);
  return (entropyA*sizeA + entropyB*sizeB) / static_cast<double>(sizeA+sizeB);
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
HeadPoseSample::makeLeaf
  (
  HeadPoseLeaf &leaf,
  const std::vector<std::shared_ptr<HeadPoseSample>> &set
  )
{
  leaf.hp_histogram.assign(upm::HP_LABELS.size(), 0);
  leaf.hp_mean = 0, leaf.hp_variance = 0;
  for (auto it_sample = set.begin(); it_sample < set.end(); ++it_sample)
  {
    float yaw = (*it_sample)->m_label.x;
    leaf.hp_histogram[upm::getHeadposeIdx(yaw)]++;
    leaf.hp_mean += yaw;
    leaf.hp_variance += (yaw*yaw);
  }
  leaf.hp_mean = (leaf.hp_mean/set.size());
  leaf.hp_variance = (leaf.hp_variance/set.size()) - (leaf.hp_mean*leaf.hp_mean);
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
double
HeadPoseSample::entropyHeadpose
  (
  const std::vector<std::shared_ptr<HeadPoseSample>> &set
  )
{
  double mean = 0, variance = 0;
  for (auto it_sample = set.begin(); it_sample < set.end(); ++it_sample)
  {
    int yaw = (*it_sample)->m_label.x;
    mean += yaw;
    variance += (yaw*yaw);
  }
  mean = (mean/set.size());
  variance = (variance/set.size()) - (mean*mean);
  return variance;
};

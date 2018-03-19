/** ****************************************************************************
 *  @file    Constants.hpp
 *  @brief   Identifiers whose associated value cannot be altered
 *  @author  Roberto Valle Fernandez
 *  @date    2015/02
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <string>
#include <vector>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <opencv2/opencv.hpp>

namespace boost {
namespace serialization {

  template<class Archive>
  void serialize(Archive &ar, cv::Size &s, const unsigned version)
  {
    ar & s.width & s.height;
  }

} // namespace serialization
} // namespace boost

/** ****************************************************************************
 * @brief Parse configuration file constants
 ******************************************************************************/
struct ForestParam
{
  int max_depth;   // tree stopping criteria
  int min_patches; // tree stopping criteria
  int ntests;      // number of tests to find the optimal split
  int nthresholds; // number of thresholds to find the optimal split
  int ntrees;      // number of trees per forest
  int stride;      // controls how densely extract patches
  int nimages;     // number of images per class
  int npatches;    // number of patches per image
  cv::Size face_size;        // face size in pixels
  cv::Size patch_size;       // patch size in pixels
  std::string tree_path;     // path to load or save the trees
  std::string image_path;    // path to load images
  std::vector<int> features; // feature channels

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & max_depth;
    ar & min_patches;
    ar & ntests;
    ar & nthresholds;
    ar & ntrees;
    ar & nimages;
    ar & npatches;
    ar & face_size;
    ar & patch_size;
    ar & tree_path;
    ar & image_path;
    ar & features;
  }
};

#endif /* CONSTANTS_HPP */

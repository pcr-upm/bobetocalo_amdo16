/** ****************************************************************************
 *  @file    ThresholdSplit.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef THRESHOLD_SPLIT_HPP
#define THRESHOLD_SPLIT_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <boost/serialization/access.hpp>
#include <boost/random/mersenne_twister.hpp>

/** ****************************************************************************
 * @class ThresholdSplit
 * @brief Split data related to a tree node
 ******************************************************************************/
template<typename Feature>
class ThresholdSplit
{
public:
  ThresholdSplit() {};

  Feature feature;
  double info;
  int threshold;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & feature;
    ar & info;
    ar & threshold;
  }
};

#endif /* THRESHOLD_SPLIT_HPP */

/** ****************************************************************************
 *  @file    ThresholdSplit.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef THRESHOLD_SPLIT_HPP
#define THRESHOLD_SPLIT_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <cereal/access.hpp>
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

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & feature;
    ar & info;
    ar & threshold;
  }
};

#endif /* THRESHOLD_SPLIT_HPP */

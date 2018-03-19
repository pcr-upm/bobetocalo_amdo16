/** ****************************************************************************
 *  @file    SplitGen.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/08
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef SPLIT_GEN_HPP
#define SPLIT_GEN_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <ThreadPool.hpp>
#include <opencv2/core/core.hpp>
#include <boost/thread.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

typedef std::pair<int, unsigned int> IntIndex;

struct less_than
{
  bool operator()(const IntIndex &a, const IntIndex &b) const
  {
    return a.first < b.first;
  };

  bool operator()(const IntIndex &a, const int &b) const
  {
    return a.first < b;
  };
};

/** ****************************************************************************
 * @class SplitGen
 * @brief Conditional regression split generator
 ******************************************************************************/
template<typename Sample>
class SplitGen
{
public:
  typedef typename Sample::Split Split;

  SplitGen
    (
    const std::vector<Sample*> &samples,
    std::vector<Split> &splits,
    boost::mt19937 *rng,
    cv::Size patch_size,
    int thresholds
    ) :
      m_samples(samples), m_splits(splits), m_patch_size(patch_size), m_thresholds(thresholds) {};

  virtual
  ~SplitGen() {};

  void
  generate()
  {
    int num_treads = boost::thread::hardware_concurrency();
    boost::thread_pool::ThreadPool e(num_treads);
    for (unsigned int stripe=0; stripe < m_splits.size(); stripe++)
      e.submit(boost::bind(&SplitGen::generateMT, this, stripe));
    e.join_all();
  };

  void
  generateMT
    (
    int stripe
    )
  {
    boost::mt19937 rng(abs(stripe+1) * std::time(NULL));
    // Random θ = (R1,R2,α)
    Sample::generateSplit(m_samples, &rng, m_patch_size, m_splits[stripe]);

    // Process each patch with the selected (R1,R2,α)
    std::vector<IntIndex> val_set(m_samples.size());
    for (unsigned int i=0; i < m_samples.size(); ++i)
    {
      val_set[i].first  = m_samples[i]->evalTest(m_splits[stripe]);
      val_set[i].second = i;
    }
    std::sort(val_set.begin(), val_set.end()); // sort by f_θ

    // Split patches with the random threshold which maximize information gain
    findThreshold(m_samples, &rng, m_thresholds, val_set, m_splits[stripe]);
  };

  static void
  splitSamples
    (
    const std::vector<Sample*> &samples,
    const std::vector<IntIndex> &val_set,
    std::vector< std::vector<Sample*> > &sets,
    int thresh
    )
  {
    // Search largest value such that value < t
    std::vector<IntIndex>::const_iterator it_middle;
    it_middle = lower_bound(val_set.begin(), val_set.end(), thresh, less_than());

    // Split training samples into two different sets A, B according to threshold t
    // No intersection between the two thresholds
    sets.resize(2);
    sets[0].resize(it_middle - val_set.begin());
    sets[1].resize(samples.size() - sets[0].size());

    std::vector<IntIndex>::const_iterator it;
    typename std::vector<Sample*>::iterator it_sample;
    it = val_set.begin();
    for (it_sample = sets[0].begin(); it_sample < sets[0].end(); ++it_sample, ++it)
      (*it_sample) = samples[it->second];

    it = val_set.begin() + sets[0].size();
    for (it_sample = sets[1].begin(); it_sample < sets[1].end(); ++it_sample, ++it)
      (*it_sample) = samples[it->second];

    /*CV_Assert((sets[0].size() + sets[1].size()) == samples.size());*/
  };

private:
  void
  findThreshold
    (
    const std::vector<Sample*> &samples,
    boost::mt19937 *rng,
    int thresholds,
    const std::vector<IntIndex> &val_set,
    Split &split
    ) const
  {
    split.info = boost::numeric::bounds<double>::highest();
    int min_val = val_set.front().first;
    int max_val = val_set.back().first;
    int range   = max_val - min_val;

    if (range > 0)
    {
      // Find best threshold
      boost::uniform_int<> dist_thr(0, range);
      boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_thr(*rng, dist_thr);

      for (int i=0; i < thresholds; ++i)
      {
        // Generate some random thresholds
        std::vector< std::vector<Sample*> > sets;
        int thresh = rand_thr() + min_val;
        splitSamples(samples, val_set, sets, thresh);

        // Each set must have more than 1 sample
        if (sets[0].size() < 2 or sets[1].size() < 2)
          continue;

        // Evaluate split using information gain
        double info = Sample::evalSplit(sets[0], sets[1]);
        if (info < split.info)
        {
          split.threshold = thresh;
          split.info = info;
        }
      }
    }
  };

  const std::vector<Sample*> &m_samples; // set of patches
  std::vector<Split> &m_splits;          // splitting candidates
  cv::Size m_patch_size;                 // patch size to generate splits
  int m_thresholds;                      // number of thresholds to test
};

#endif /* SPLIT_GEN_HPP */

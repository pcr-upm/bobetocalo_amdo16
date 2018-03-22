/** ****************************************************************************
 *  @file    FaceHeadPoseForest.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <utils.hpp>
#include <FaceHeadPoseForest.hpp>
#include <boost/progress.hpp>
#include <boost/program_options.hpp>
#include <boost/multi_array.hpp>

namespace upm {

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
FaceHeadPoseForest::FaceHeadPoseForest(std::string path)
{
  ForestParam hp_param;
  hp_param.tree_path = path;
  hp_param.ntrees = 20;
  hp_param.stride = 10;
  hp_param.ntests = 2000;
  hp_param.nthresholds = 25;
  hp_param.max_depth = 15;
  hp_param.min_patches = 20;
  hp_param.nimages = 2000;
  hp_param.npatches = 20;
  hp_param.face_size = cv::Size(125,125);
  hp_param.patch_size = cv::Size(61,61);
  hp_param.features = {0, 1, 2};
  hp_forest_param = hp_param;
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
FaceHeadPoseForest::parseOptions
  (
  int argc,
  char **argv
  )
{
  /// Declare the supported program options
  namespace po = boost::program_options;
  po::options_description desc("FaceHeadPoseForest options");
  desc.add_options()
    ("tree", po::value<int>(), "Choose tree index for simultaneous train");
  UPM_PRINT(desc);

  /// Process the command line parameters
  po::variables_map vm;
  po::command_line_parser parser(argc, argv);
  parser.options(desc);
  const po::parsed_options parsed_opt(parser.allow_unregistered().run());
  po::store(parsed_opt, vm);
  po::notify(vm);

  if (vm.count("tree"))
    hp_trees_idx.push_back(vm["tree"].as<int>());
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
FaceHeadPoseForest::train
  (
  const std::vector<upm::FaceAnnotation> &anns_train,
  const std::vector<upm::FaceAnnotation> &anns_valid
  )
{
  std::vector<upm::FaceAnnotation> anns;
  anns.insert(anns.end(), anns_train.begin(), anns_train.end());
  anns.insert(anns.end(), anns_valid.begin(), anns_valid.end());
  unsigned int num_train = static_cast<unsigned int>(anns_train.size());
  /// Divide annotations by head-pose classes
  const unsigned int num_headposes = static_cast<int>(HP_LABELS.size());
  boost::multi_array<std::vector<upm::FaceAnnotation>,3> hp_anns(boost::extents[num_headposes][1][1]);
  for (const FaceAnnotation &ann : anns)
  {
    int yaw_idx = getHeadposeIdx(getHeadpose(ann).x);
    hp_anns[yaw_idx][0][0].push_back(ann);
  }

  /// Balanced number of images for each class
  unsigned int imgs_per_class = static_cast<unsigned int>(hp_forest_param.nimages);
  for (auto it=hp_anns.origin(); it < hp_anns.origin()+hp_anns.num_elements(); it++)
    if (not (*it).empty())
      imgs_per_class =  static_cast<int>(MIN(imgs_per_class, (*it).size()));

  /// Default decision-tree indices (0..19) for train
  if (hp_trees_idx.empty())
  {
    hp_trees_idx.resize(hp_forest_param.ntrees);
    std::iota(hp_trees_idx.begin(), hp_trees_idx.end(), 0);
  }
  for (int tree_idx: hp_trees_idx)
  {
    /// Try to read the head-pose regression tree
    char tree_path[200];
    sprintf(tree_path, "%stree_%03d.bin", hp_forest_param.tree_path.c_str(), tree_idx);
    UPM_PRINT("Read head pose regression tree: " << tree_path);
    std::shared_ptr<Tree<HeadPoseSample>> tree;
    bool is_tree_load = Tree<HeadPoseSample>::load(tree, tree_path);

    /// Select random annotations for this head-pose
    std::vector<FaceAnnotation> rnd_data;
    for (auto it=hp_anns.origin(); it < hp_anns.origin()+hp_anns.num_elements(); it++)
      if (not (*it).empty())
        rnd_data.insert(rnd_data.end(), (*it).begin(), (*it).begin()+imgs_per_class);
    UPM_PRINT("Number of images per class: " << imgs_per_class);
    UPM_PRINT("Total number of images: " << rnd_data.size());

    std::vector<std::shared_ptr<HeadPoseSample>> hp_samples;
    hp_samples.reserve(rnd_data.size()*hp_forest_param.npatches);
    UPM_PRINT("Reserved patches: " << rnd_data.size()*hp_forest_param.npatches);

    boost::mt19937 rng;
    rng.seed(tree_idx+1);
    boost::progress_display show_progress(rnd_data.size());
    for (int i=0; i < rnd_data.size(); i++, ++show_progress)
    {
      /// Load image
      cv::Mat frame = cv::imread(rnd_data[i].filename, cv::IMREAD_COLOR);
      if (frame.empty())
        continue;

      /// Convert image to gray scale
      cv::Mat img_gray;
      cv::cvtColor(frame, img_gray, cv::COLOR_BGR2GRAY);

      /// Scale image
      cv::Mat face_translated, T = (cv::Mat_<float>(2,3) << 1, 0, -rnd_data[i].bbox.pos.x, 0, 1, -rnd_data[i].bbox.pos.y);
      cv::warpAffine(img_gray, face_translated, T, img_gray.size());
      cv::Mat face_scaled, S = (cv::Mat_<float>(2,3) << hp_forest_param.face_size.width/rnd_data[i].bbox.pos.width, 0, 0, 0, hp_forest_param.face_size.height/rnd_data[i].bbox.pos.height, 0);
      cv::warpAffine(face_translated, face_scaled, S, hp_forest_param.face_size);

      /// Extract patches from this face sample
      boost::shared_ptr<ImageSample> sample(new ImageSample(face_scaled, hp_forest_param.features, false));

      /// Generate random patches
      boost::uniform_int<> dist_x(0, face_scaled.cols-hp_forest_param.patch_size.width-1);
      boost::uniform_int<> dist_y(0, face_scaled.rows-hp_forest_param.patch_size.height-1);
      boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_x(rng, dist_x);
      boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_y(rng, dist_y);
      for (int j=0; j < hp_forest_param.npatches; j++)
      {
        cv::Rect bbox_patch = cv::Rect(rand_x(), rand_y(), hp_forest_param.patch_size.width, hp_forest_param.patch_size.height);
        std::shared_ptr<HeadPoseSample> hps(new HeadPoseSample(sample, bbox_patch, rnd_data[i].headpose));
        hp_samples.push_back(hps);
//        hps->show();
      }
    }
    UPM_PRINT("Used patches: " << hp_samples.size());

    if (is_tree_load and (not tree->isFinished()))
      tree->update(hp_samples, &rng);
    else
      tree = std::make_shared<Tree<HeadPoseSample>>(Tree<HeadPoseSample>(hp_samples, hp_forest_param, &rng, tree_path));
  }
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
FaceHeadPoseForest::load()
{
  /// Loading head-pose classifier
  UPM_PRINT("Loading head-pose forest");
  if (not hp_forest.load(hp_forest_param.tree_path, hp_forest_param))
    UPM_ERROR("Error loading head-pose forest");
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
FaceHeadPoseForest::process
  (
  cv::Mat frame,
  std::vector<upm::FaceAnnotation> &faces,
  const upm::FaceAnnotation &ann
  )
{
  /// Convert image to gray scale
  cv::Mat img_gray;
  cv::cvtColor(frame, img_gray, cv::COLOR_BGR2GRAY);

  /// Analyze each detected face
  for (FaceAnnotation &face : faces)
  {
    /// Scale image
    cv::Mat face_translated, T = (cv::Mat_<float>(2,3) << 1, 0, -face.bbox.pos.x, 0, 1, -face.bbox.pos.y);
    cv::warpAffine(img_gray, face_translated, T, img_gray.size());
    cv::Mat face_scaled, S = (cv::Mat_<float>(2,3) << hp_forest_param.face_size.width/face.bbox.pos.width, 0, 0, 0, hp_forest_param.face_size.height/face.bbox.pos.height, 0);
    cv::warpAffine(face_translated, face_scaled, S, hp_forest_param.face_size);

    /// Extract patches from this face sample
    boost::shared_ptr<ImageSample> sample(new ImageSample(face_scaled, hp_forest_param.features, true));

    /// Store head-pose angle
    face.headpose = getHeadPoseVotesMT(sample, hp_forest);
  }
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
cv::Point3f
FaceHeadPoseForest::getHeadPoseVotesMT
  (
  boost::shared_ptr<ImageSample> sample,
  const Forest<HeadPoseSample> &forest
  )
{
  /// Reserve patches like a dense extraction for stride 1
  int width = sample->m_feature_channels[0].cols - forest.getParam().patch_size.width;
  int height = sample->m_feature_channels[0].rows - forest.getParam().patch_size.height;
  std::vector<std::shared_ptr<HeadPoseSample>> samples;
  samples.reserve(width * height);
  for (int x=0; x < width; x += hp_forest_param.stride)
    for (int y=0; y < height; y += hp_forest_param.stride)
    {
      cv::Rect bbox_patch(x, y, forest.getParam().patch_size.width, forest.getParam().patch_size.height);
      samples.push_back(std::make_shared<HeadPoseSample>(HeadPoseSample(sample, bbox_patch)));
    }

  /// Process each patch
  boost::thread_pool::ThreadPool e(boost::thread::hardware_concurrency());
  int num_trees = forest.numberOfTrees();
  std::vector<std::shared_ptr<HeadPoseLeaf>> leafs;
  leafs.resize(samples.size() * num_trees);
  for (unsigned int i=0; i < samples.size(); i++)
    e.submit(boost::bind(&Forest<HeadPoseSample>::evaluateMT, forest, samples[i], &leafs[i*num_trees]));
  e.join_all();

  /// Parse collected leafs
  const float MAX_VARIANCE = 400.0f;
  std::vector<int> histogram(HP_LABELS.size(), 0);
  for (std::shared_ptr<HeadPoseLeaf> leaf : leafs)
  {
    /// Filter out leaves with a high variance which are less informative
    if (leaf->hp_variance > MAX_VARIANCE)
      continue;

    for (unsigned int j=0; j < leaf->hp_histogram.size(); j++)
      histogram[j] += leaf->hp_histogram[j];
  }

  /// Classification
  int yaw_idx = static_cast<int>(std::distance(histogram.begin(), std::max_element(histogram.begin(), histogram.end())));
  return cv::Point3f(HP_LABELS[yaw_idx],0,0);
};

} // namespace upm

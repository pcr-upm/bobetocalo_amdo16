/** ****************************************************************************
 *  @file    FeatureChannelFactory.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/09
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FEATURE_CHANNEL_FACTORY_HPP
#define FEATURE_CHANNEL_FACTORY_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <ThreadPool.hpp>
#include <stdexcept>
#include <boost/thread.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define FC_GRAY    0
#define FC_GABOR   1
#define FC_SOBEL   2
#define FC_MIN_MAX 3
#define FC_CANNY   4
#define FC_NORM    5

/** ****************************************************************************
 * @class FeatureChannelFactory
 * @brief Feature channel extractor
 ******************************************************************************/
class FeatureChannelFactory
{
public:
  FeatureChannelFactory() {};

  ~FeatureChannelFactory() {};

  void
  extractChannel
    (
    int feature,
    bool use_integral,
    const cv::Mat &img,
    std::vector<cv::Mat> &channels
    )
  {
    switch (feature)
    {
      case FC_GRAY:
      {
        channels.push_back(integralImage(use_integral, img));
        break;
      }
      case FC_GABOR:
      {
        // Check if kernels are initialized
        if (reals.size() == 0)
          initGaborKernels();

        unsigned int sz = channels.size();
        channels.resize(sz+reals.size());
        int num_treads = boost::thread::hardware_concurrency();
        boost::thread_pool::ThreadPool e(num_treads);
        for (unsigned int i=0; i < reals.size(); i++)
          e.submit(boost::bind(&FeatureChannelFactory::gaborTransform, this, use_integral, img, i, &channels[sz+i]));
        e.join_all();
        break;
      }
      case FC_SOBEL:
      {
        cv::Mat sobx_img(img.size(), CV_8U);
        cv::Mat soby_img(img.size(), CV_8U);
        cv::Sobel(img, sobx_img, CV_8U, 0, 1);
        cv::Sobel(img, soby_img, CV_8U, 1, 0);
        channels.push_back(integralImage(use_integral, sobx_img));
        channels.push_back(integralImage(use_integral, soby_img));
        break;
      }
      case FC_MIN_MAX:
      {
        cv::Mat kernel(cv::Size(3, 3), CV_8UC1);
        kernel.setTo(cv::Scalar(1));
        cv::Mat min_img(img.size(), CV_8U);
        cv::Mat max_img(img.size(), CV_8U);
        cv::erode(img, min_img, kernel);
        cv::dilate(img, max_img, kernel);
        channels.push_back(integralImage(use_integral, min_img));
        channels.push_back(integralImage(use_integral, max_img));
        break;
      }
      case FC_CANNY:
      {
        cv::Mat canny_img;
        cv::Canny(img, canny_img, -1, 5);
        channels.push_back(integralImage(use_integral, canny_img));
        break;
      }
      case FC_NORM:
      {
        cv::Mat equalize_img;
        cv::equalizeHist(img, equalize_img);
        channels.push_back(integralImage(use_integral, equalize_img));
        break;
      }
      default:
        throw std::invalid_argument("unknown feature channel to extract");
    }
  };

private:
  cv::Mat
  integralImage
    (
    bool use_integral,
    const cv::Mat &img
    ) const
  {
    if (not use_integral)
      return img;

    cv::Mat integral_img;
    cv::integral(img, integral_img, CV_32F);
    return integral_img;
  };

  void
  gaborTransform
    (
    bool use_integral,
    const cv::Mat &src,
    int index,
    cv::Mat *dst
    ) const
  {
    cv::Mat final;
    cv::Mat r_mat;
    cv::Mat i_mat;
    cv::filter2D(src, r_mat, CV_32F, reals[index]);
    cv::filter2D(src, i_mat, CV_32F, imags[index]);
    cv::pow(r_mat, 2, r_mat);
    cv::pow(i_mat, 2, i_mat);
    cv::add(i_mat, r_mat, final);
    cv::pow(final, 0.5, final);
    cv::normalize(final, final, 0, 1, cv::NORM_MINMAX);

    cv::Mat img;
    final.convertTo(img, CV_8UC1, 255);
    *dst = integralImage(use_integral, img);
  };

  void
  createKernel
    (
    int iMu,
    int iNu,
    double sigma,
    double dF
    )
  {
    // Initialize the parameters
    double F = dF;
    double k = (CV_PI / 2) / pow(F, (double) iNu);
    double phi = CV_PI * iMu / 8;

    double width = round((sigma / k) * 6 + 1);
    if (fmod(width, 2.0) == 0.0)
      width++;

    // Create kernel
    cv::Mat real = cv::Mat(width, width, CV_32FC1);
    cv::Mat imag = cv::Mat(width, width, CV_32FC1);

    int x, y;
    double dReal;
    double dImag;
    double dTemp1, dTemp2, dTemp3;

    int off_set = (width - 1) / 2;
    for (int i = 0; i < width; i++)
    {
      for (int j = 0; j < width; j++)
      {
        x = i - off_set;
        y = j - off_set;
        dTemp1 = (pow(k, 2) / pow(sigma, 2)) * exp(-(pow((double) x, 2) + pow((double) y, 2)) * pow(k, 2) / (2 * pow(sigma, 2)));
        dTemp2 = cos(k * cos(phi) * x + k * sin(phi) * y) - exp(-(pow(sigma, 2) / 2));
        dTemp3 = sin(k * cos(phi) * x + k * sin(phi) * y);
        dReal = dTemp1 * dTemp2;
        dImag = dTemp1 * dTemp3;
        real.at<float>(j, i) = dReal;
        imag.at<float>(j, i) = dImag;
      }
    }

    reals.push_back(real);
    imags.push_back(imag);
  };

  void
  initGaborKernels()
  {
    // Create kernels
    int NuMin = 0;
    int NuMax = 4;
    int MuMin = 0;
    int MuMax = 7;
    double sigma = 1.0 / 2.0 * CV_PI;
    double dF = sqrt(2.0);

    int iMu = 0;
    int iNu = 0;

    for (iNu = NuMin; iNu <= NuMax; iNu++)
      for (iMu = MuMin; iMu < MuMax; iMu++)
        createKernel(iMu, iNu, sigma, dF);
  };

  std::vector<cv::Mat> reals; // real
  std::vector<cv::Mat> imags; // imaginary
};

#endif /* FEATURECHANNELEXTRACTOR_H_ */

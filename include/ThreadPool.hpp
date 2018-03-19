/** ****************************************************************************
 *  @file    ThreadPool.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/07
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

namespace boost {
namespace thread_pool {

/** ****************************************************************************
 * @class ThreadPool
 * @brief Use boost threads and io_service to create a thread pool
 ******************************************************************************/
class ThreadPool
{
public:
  ThreadPool
    (
    size_t n = 10
    ) : m_service(n), m_worker(new boost::asio::io_service::work(m_service))
  {
    for (size_t i=0; i < n; i++)
      m_pool.create_thread(boost::bind(&boost::asio::io_service::run, &m_service));
  };

  ~ThreadPool()
  {
    m_worker.reset();
    m_service.stop();
    m_pool.join_all();
  };

  void
  join_all()
  {
    m_worker.reset();
    m_pool.join_all();
  };

  template<typename F>
  void
  submit
    (
    F task
    )
  {
    m_service.post(task);
  };

protected:
  boost::thread_group m_pool;
  boost::asio::io_service m_service;
  boost::shared_ptr<boost::asio::io_service::work> m_worker;
};

} // thread_pool
} // boost

#endif /* THREAD_POOL_HPP */

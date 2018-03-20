/** ****************************************************************************
 *  @file    TreeNode.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef TREE_NODE_HPP
#define TREE_NODE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <vector>
#include <cereal/access.hpp>

/** ****************************************************************************
 * @class TreeNode
 * @brief Conditional regression tree node
 ******************************************************************************/
template<typename Sample>
class TreeNode
{
public:
  typedef typename Sample::Split Split;
  typedef typename Sample::Leaf Leaf;

  TreeNode() :
    depth(-1), right(NULL), left(NULL), is_leaf(false), is_split(false) {};

  TreeNode(int d) :
    depth(d), right(NULL), left(NULL), is_leaf(false), is_split(false) {};

  ~TreeNode() {};

  int
  getDepth()
  {
    return depth;
  };

  bool
  isLeaf() const
  {
    return is_leaf;
  };

  std::shared_ptr<Leaf>
  getLeaf()
  {
    return std::make_shared<Leaf>(leaf);
  };

  void
  createLeaf
    (
    const std::vector<std::shared_ptr<Sample>> &samples
    )
  {
    Sample::makeLeaf(leaf, samples);
    is_leaf = true;
    is_split = false;
  };

  bool
  isSplit() const
  {
    return is_split;
  };

  Split
  getSplit()
  {
    return split;
  };

  void
  createSplit
    (
    Split s
    )
  {
    is_leaf = false;
    is_split = true;
    split = s;
  };

  void
  addLeftChild
    (
    std::shared_ptr<TreeNode<Sample>> left_child
    )
  {
    left = left_child;
  };

  void
  addRightChild
    (
    std::shared_ptr<TreeNode<Sample>> right_child
    )
  {
    right = right_child;
  };

  bool
  eval
    (
    const std::shared_ptr<Sample> s
    ) const
  {
    return s->eval(split);
  };

  Leaf leaf;
  Split split;
  std::shared_ptr<TreeNode<Sample>> right;
  std::shared_ptr<TreeNode<Sample>> left;

private:
  int depth;
  bool is_leaf;
  bool is_split;

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & depth;
    ar & is_leaf;
    ar & is_split;
    if (is_leaf)
      ar & leaf;
    if (is_split)
      ar & split;
    if (!is_leaf)
    {
      ar & left;
      ar & right;
    }
  }
};

#endif /* TREE_NODE_HPP */

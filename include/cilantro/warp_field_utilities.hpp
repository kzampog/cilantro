#pragma once

#include <cilantro/space_transformations.hpp>
#include <cilantro/common_pair_evaluators.hpp>
#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, class WeightEvaluatorT = UnityWeightEvaluator<ScalarT,ScalarT>>
    void resampleTransformations(const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                 const std::vector<NeighborSet<typename WeightEvaluatorT::InputScalar>> &new_to_old_map,
                                 RigidTransformationSet<ScalarT,EigenDim> &new_transforms,
                                 const WeightEvaluatorT &weight_evaluator = WeightEvaluatorT())
    {
        new_transforms.resize(new_to_old_map.size());

#pragma omp parallel for shared (new_transforms)
        for (size_t i = 0; i < new_transforms.size(); i++) {
            ScalarT total_weight = (ScalarT)0.0;
            new_transforms[i].linear().setZero();
            new_transforms[i].translation().setZero();
            for (size_t j = 0; j < new_to_old_map[i].size(); j++) {
                const ScalarT weight = weight_evaluator(i, new_to_old_map[i][j].index, new_to_old_map[i][j].value);
                total_weight += weight;
                new_transforms[i].linear().noalias() += weight*old_transforms[new_to_old_map[i][j].index].linear();
                new_transforms[i].translation().noalias() += weight*old_transforms[new_to_old_map[i][j].index].translation();
            }

            if (total_weight == (ScalarT)0.0) {
                new_transforms[i].setIdentity();
            } else {
                total_weight = (ScalarT)(1.0)/total_weight;
                new_transforms[i].linear() *= total_weight;
                new_transforms[i].linear() = new_transforms[i].rotation();
                new_transforms[i].translation() *= total_weight;
            }
        }
    }

    template <typename ScalarT, ptrdiff_t EigenDim, class WeightEvaluatorT = UnityWeightEvaluator<ScalarT,ScalarT>>
    inline RigidTransformationSet<ScalarT,EigenDim> resampleTransformations(const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                                                            const std::vector<NeighborSet<typename WeightEvaluatorT::InputScalar>> &new_to_old_map,
                                                                            const WeightEvaluatorT &weight_evaluator = WeightEvaluatorT())
    {
        RigidTransformationSet<ScalarT,EigenDim> new_transforms;
        resampleTransformations<ScalarT,EigenDim,WeightEvaluatorT>(old_transforms, new_to_old_map, new_transforms, weight_evaluator);
        return new_transforms;
    }

    template <typename ScalarT, ptrdiff_t EigenDim, NeighborhoodType NT, class WeightEvaluatorT = UnityWeightEvaluator<ScalarT,ScalarT>>
    void resampleTransformations(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &old_support_kd_tree,
                                 const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                 const ConstVectorSetMatrixMap<ScalarT,EigenDim> &new_support,
                                 const NeighborhoodSpecification<ScalarT> &nh,
                                 RigidTransformationSet<ScalarT,EigenDim> &new_transforms,
                                 const WeightEvaluatorT &weight_evaluator = WeightEvaluatorT())
    {
        new_transforms.resize(new_support.cols());

        NeighborSet<ScalarT> nn;
#pragma omp parallel for shared (new_transforms) private (nn)
        for (size_t i = 0; i < new_transforms.size(); i++) {
            old_support_kd_tree.template search<NT>(new_support.col(i), nh, nn);
            ScalarT total_weight = (ScalarT)0.0;
            new_transforms[i].linear().setZero();
            new_transforms[i].translation().setZero();
            for (size_t j = 0; j < nn.size(); j++) {
                const ScalarT weight = weight_evaluator(i, nn[j].index, nn[j].value);
                total_weight += weight;
                new_transforms[i].linear().noalias() += weight*old_transforms[nn[j].index].linear();
                new_transforms[i].translation().noalias() += weight*old_transforms[nn[j].index].translation();
            }

            if (total_weight == (ScalarT)0.0) {
                new_transforms[i].setIdentity();
            } else {
                total_weight = (ScalarT)(1.0)/total_weight;
                new_transforms[i].linear() *= total_weight;
                new_transforms[i].linear() = new_transforms[i].rotation();
                new_transforms[i].translation() *= total_weight;
            }
        }
    }

    template <typename ScalarT, ptrdiff_t EigenDim, NeighborhoodType NT, class WeightEvaluatorT = UnityWeightEvaluator<ScalarT,ScalarT>>
    inline RigidTransformationSet<ScalarT,EigenDim> resampleTransformations(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &old_support_kd_tree,
                                                                            const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                                                            const ConstVectorSetMatrixMap<ScalarT,EigenDim> &new_support,
                                                                            const NeighborhoodSpecification<ScalarT> &nh,
                                                                            const WeightEvaluatorT &weight_evaluator = WeightEvaluatorT())
    {
        RigidTransformationSet<ScalarT,EigenDim> new_transforms;
        resampleTransformations<ScalarT,EigenDim,NT,WeightEvaluatorT>(old_support_kd_tree, old_transforms, new_support, nh, new_transforms, weight_evaluator);
        return new_transforms;
    }

    template <typename ScalarT, ptrdiff_t EigenDim, class WeightEvaluatorT = UnityWeightEvaluator<ScalarT,ScalarT>>
    void resampleTransformations(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &old_support_kd_tree,
                                 const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                 const ConstVectorSetMatrixMap<ScalarT,EigenDim> &new_support,
                                 const NeighborhoodSpecification<ScalarT> &nh,
                                 RigidTransformationSet<ScalarT,EigenDim> &new_transforms,
                                 const WeightEvaluatorT &weight_evaluator = WeightEvaluatorT())
    {
        switch (nh.type) {
            case NeighborhoodType::KNN:
                resampleTransformations<ScalarT,EigenDim,NeighborhoodType::KNN,WeightEvaluatorT>(old_support_kd_tree, old_transforms, new_support, nh, new_transforms, weight_evaluator);
                break;
            case NeighborhoodType::RADIUS:
                resampleTransformations<ScalarT,EigenDim,NeighborhoodType::RADIUS,WeightEvaluatorT>(old_support_kd_tree, old_transforms, new_support, nh, new_transforms, weight_evaluator);
                break;
            case NeighborhoodType::KNN_IN_RADIUS:
                resampleTransformations<ScalarT,EigenDim,NeighborhoodType::KNN_IN_RADIUS,WeightEvaluatorT>(old_support_kd_tree, old_transforms, new_support, nh, new_transforms, weight_evaluator);
                break;
        }
    }

    template <typename ScalarT, ptrdiff_t EigenDim, class WeightEvaluatorT = UnityWeightEvaluator<ScalarT,ScalarT>>
    inline RigidTransformationSet<ScalarT,EigenDim> resampleTransformations(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &old_support_kd_tree,
                                                                            const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                                                            const ConstVectorSetMatrixMap<ScalarT,EigenDim> &new_support,
                                                                            const NeighborhoodSpecification<ScalarT> &nh,
                                                                            const WeightEvaluatorT &weight_evaluator = WeightEvaluatorT())
    {
        RigidTransformationSet<ScalarT,EigenDim> new_transforms;
        resampleTransformations<ScalarT,EigenDim,WeightEvaluatorT>(old_support_kd_tree, old_transforms, new_support, nh, new_transforms, weight_evaluator);
        return new_transforms;
    }
}

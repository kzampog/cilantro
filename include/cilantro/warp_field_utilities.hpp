#pragma once

#include <cilantro/space_transformations.hpp>
#include <cilantro/kd_tree.hpp>

namespace cilantro {
    // Values interpreted as weights
    template <typename ScalarT, ptrdiff_t EigenDim>
    void resampleTransformations(const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                 const std::vector<NeighborSet<ScalarT>> &new_to_old_map,
                                 RigidTransformationSet<ScalarT,EigenDim> &new_transforms)
    {
        new_transforms.resize(new_to_old_map.size());

        ScalarT total_weight;

#pragma omp parallel for shared (new_transforms) private (total_weight)
        for (size_t i = 0; i < new_transforms.size(); i++) {
            total_weight = (ScalarT)0.0;
            new_transforms[i].linear().setZero();
            new_transforms[i].translation().setZero();
            for (size_t j = 0; j < new_to_old_map[i].size(); j++) {
                total_weight += new_to_old_map[i][j].value;
                new_transforms[i].linear().noalias() += new_to_old_map[i][j].value*old_transforms[new_to_old_map[i][j].index].linear();
                new_transforms[i].translation().noalias() += new_to_old_map[i][j].value*old_transforms[new_to_old_map[i][j].index].translation();
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

    // Values interpreted as weights
    template <typename ScalarT, ptrdiff_t EigenDim>
    inline RigidTransformationSet<ScalarT,EigenDim> resampleTransformations(const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                                                            const std::vector<NeighborSet<ScalarT>> &new_to_old_map)
    {
        RigidTransformationSet<ScalarT,EigenDim> new_transforms;
        resampleTransformations<ScalarT,EigenDim>(old_transforms, new_to_old_map, new_transforms);
        return new_transforms;
    }

    // Values interpreted as distances
    template <typename ScalarT, ptrdiff_t EigenDim>
    void resampleTransformations(const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                 const std::vector<NeighborSet<ScalarT>> &new_to_old_map,
                                 ScalarT distance_sigma,
                                 RigidTransformationSet<ScalarT,EigenDim> &new_transforms)
    {
        new_transforms.resize(new_to_old_map.size());

        const ScalarT sigma_inv_sq = (ScalarT)(1.0)/(distance_sigma*distance_sigma);
        ScalarT curr_weight, total_weight;

#pragma omp parallel for shared (new_transforms) private (curr_weight, total_weight)
        for (size_t i = 0; i < new_transforms.size(); i++) {
            total_weight = (ScalarT)0.0;
            new_transforms[i].linear().setZero();
            new_transforms[i].translation().setZero();
            for (size_t j = 0; j < new_to_old_map[i].size(); j++) {
                curr_weight = std::exp(-(ScalarT)(0.5)*new_to_old_map[i][j].value*sigma_inv_sq);
                total_weight += curr_weight;
                new_transforms[i].linear().noalias() += curr_weight*old_transforms[new_to_old_map[i][j].index].linear();
                new_transforms[i].translation().noalias() += curr_weight*old_transforms[new_to_old_map[i][j].index].translation();
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

    // Values interpreted as distances
    template <typename ScalarT, ptrdiff_t EigenDim>
    inline RigidTransformationSet<ScalarT,EigenDim> resampleTransformations(const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                                                            const std::vector<NeighborSet<ScalarT>> &new_to_old_map,
                                                                            ScalarT distance_sigma)
    {
        RigidTransformationSet<ScalarT,EigenDim> new_transforms;
        resampleTransformations<ScalarT,EigenDim>(old_transforms, new_to_old_map, distance_sigma, new_transforms);
        return new_transforms;
    }

    template <typename ScalarT, ptrdiff_t EigenDim, NeighborhoodType NT>
    void resampleTransformations(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &old_support_kd_tree,
                                 const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                 const ConstVectorSetMatrixMap<ScalarT,EigenDim> &new_support,
                                 const NeighborhoodSpecification<ScalarT> &nh,
                                 ScalarT distance_sigma,
                                 RigidTransformationSet<ScalarT,EigenDim> &new_transforms)
    {
        new_transforms.resize(new_support.cols());
        const ScalarT sigma_inv_sq = (ScalarT)(1.0)/(distance_sigma*distance_sigma);

        NeighborSet<ScalarT> nn;
        ScalarT curr_weight, total_weight;

#pragma omp parallel for shared (new_transforms) private (nn, curr_weight, total_weight)
        for (size_t i = 0; i < new_transforms.size(); i++) {
            old_support_kd_tree.template search<NT>(new_support.col(i), nh, nn);

            total_weight = (ScalarT)0.0;
            new_transforms[i].linear().setZero();
            new_transforms[i].translation().setZero();
            for (size_t j = 0; j < nn.size(); j++) {
                curr_weight = std::exp(-(ScalarT)(0.5)*nn[j].value*sigma_inv_sq);
                total_weight += curr_weight;
                new_transforms[i].linear().noalias() += curr_weight*old_transforms[nn[j].index].linear();
                new_transforms[i].translation().noalias() += curr_weight*old_transforms[nn[j].index].translation();
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

    template <typename ScalarT, ptrdiff_t EigenDim, NeighborhoodType NT>
    inline RigidTransformationSet<ScalarT,EigenDim> resampleTransformations(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &old_support_kd_tree,
                                                                            const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                                                            const ConstVectorSetMatrixMap<ScalarT,EigenDim> &new_support,
                                                                            const NeighborhoodSpecification<ScalarT> &nh,
                                                                            ScalarT distance_sigma)
    {
        RigidTransformationSet<ScalarT,EigenDim> new_transforms;
        resampleTransformations<ScalarT,EigenDim,NT>(old_support_kd_tree, old_transforms, new_support, nh, distance_sigma, new_transforms);
        return new_transforms;
    }

    template <typename ScalarT, ptrdiff_t EigenDim>
    void resampleTransformations(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &old_support_kd_tree,
                                 const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                 const ConstVectorSetMatrixMap<ScalarT,EigenDim> &new_support,
                                 const NeighborhoodSpecification<ScalarT> &nh,
                                 ScalarT distance_sigma,
                                 RigidTransformationSet<ScalarT,EigenDim> &new_transforms)
    {
        switch (nh.type) {
            case NeighborhoodType::KNN:
                resampleTransformations<ScalarT,EigenDim,NeighborhoodType::KNN>(old_support_kd_tree, old_transforms, new_support, nh, distance_sigma, new_transforms);
                break;
            case NeighborhoodType::RADIUS:
                resampleTransformations<ScalarT,EigenDim,NeighborhoodType::RADIUS>(old_support_kd_tree, old_transforms, new_support, nh, distance_sigma, new_transforms);
                break;
            case NeighborhoodType::KNN_IN_RADIUS:
                resampleTransformations<ScalarT,EigenDim,NeighborhoodType::KNN_IN_RADIUS>(old_support_kd_tree, old_transforms, new_support, nh, distance_sigma, new_transforms);
                break;
        }
    }

    template <typename ScalarT, ptrdiff_t EigenDim>
    inline RigidTransformationSet<ScalarT,EigenDim> resampleTransformations(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &old_support_kd_tree,
                                                                            const RigidTransformationSet<ScalarT,EigenDim> &old_transforms,
                                                                            const ConstVectorSetMatrixMap<ScalarT,EigenDim> &new_support,
                                                                            const NeighborhoodSpecification<ScalarT> &nh,
                                                                            ScalarT distance_sigma)
    {
        RigidTransformationSet<ScalarT,EigenDim> new_transforms;
        resampleTransformations<ScalarT,EigenDim>(old_support_kd_tree, old_transforms, new_support, nh, distance_sigma, new_transforms);
        return new_transforms;
    }
}

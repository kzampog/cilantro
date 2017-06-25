#include <cilantro/rigid_transform_estimator.hpp>
#include <cilantro/transform_estimation.hpp>

RigidTransformEstimator::RigidTransformEstimator(const std::vector<Eigen::Vector3f> &dst_points,
                                                 const std::vector<Eigen::Vector3f> &src_points)
        : RandomSampleConsensus(3, 50, 100, 0.01, true),
          dst_points_(&dst_points),
          src_points_(&src_points)
{}

RigidTransformEstimator::RigidTransformEstimator(const std::vector<Eigen::Vector3f> &dst_points,
                                                 const std::vector<Eigen::Vector3f> &src_points,
                                                 const std::vector<size_t> &dst_ind,
                                                 const std::vector<size_t> &src_ind)
        : RandomSampleConsensus(3, 50, 100, 0.01, true),
          dst_points_tmp_(std::vector<Eigen::Vector3f>(dst_ind.size())),
          src_points_tmp_(std::vector<Eigen::Vector3f>(src_ind.size())),
          dst_points_(&dst_points_tmp_),
          src_points_(&src_points_tmp_)
{
    for (size_t i = 0; i < dst_ind.size(); i++) {
        dst_points_tmp_[i] = dst_points[dst_ind[i]];
        src_points_tmp_[i] = src_points[src_ind[i]];
    }
}

RigidTransformEstimator::RigidTransformEstimator(const PointCloud &dst,
                                                 const PointCloud &src)
        : RandomSampleConsensus(3, 50, 100, 0.01, true),
          dst_points_(&dst.points),
          src_points_(&src.points)
{}

RigidTransformEstimator::RigidTransformEstimator(const PointCloud &dst,
                                                 const PointCloud &src,
                                                 const std::vector<size_t> &dst_ind,
                                                 const std::vector<size_t> &src_ind)
        : RandomSampleConsensus(3, 50, 100, 0.01, true),
          dst_points_tmp_(std::vector<Eigen::Vector3f>(dst_ind.size())),
          src_points_tmp_(std::vector<Eigen::Vector3f>(src_ind.size())),
          dst_points_(&dst_points_tmp_),
          src_points_(&src_points_tmp_)
{
    for (size_t i = 0; i < dst_ind.size(); i++) {
        dst_points_tmp_[i] = dst.points[dst_ind[i]];
        src_points_tmp_[i] = src.points[src_ind[i]];
    }
}

RigidTransformEstimator& RigidTransformEstimator::estimateModelParameters(RigidTransformParameters &model_params) {
    estimateRigidTransformPointToPoint(*dst_points_, *src_points_, model_params.rotation, model_params.translation);
    return *this;
}

RigidTransformParameters RigidTransformEstimator::estimateModelParameters() {
    RigidTransformParameters model_params;
    estimateModelParameters(model_params);
    return model_params;
}

RigidTransformEstimator& RigidTransformEstimator::estimateModelParameters(const std::vector<size_t> &sample_ind, RigidTransformParameters &model_params) {
    std::vector<Eigen::Vector3f> dst_p(sample_ind.size());
    std::vector<Eigen::Vector3f> src_p(sample_ind.size());
    for (size_t i = 0; i < sample_ind.size(); i++) {
        dst_p[i] = (*dst_points_)[sample_ind[i]];
        src_p[i] = (*src_points_)[sample_ind[i]];
    }
    estimateRigidTransformPointToPoint(dst_p, src_p, model_params.rotation, model_params.translation);
    return *this;
}

RigidTransformParameters RigidTransformEstimator::estimateModelParameters(const std::vector<size_t> &sample_ind) {
    RigidTransformParameters model_params;
    estimateModelParameters(sample_ind, model_params);
    return model_params;
}

RigidTransformEstimator& RigidTransformEstimator::computeResiduals(const RigidTransformParameters &model_params, std::vector<float> &residuals) {
    residuals.resize(dst_points_->size());
    Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > dst((float *)dst_points_->data(),3,dst_points_->size());
    Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > src((float *)src_points_->data(),3,src_points_->size());
    Eigen::Map<Eigen::Matrix<float,1,Eigen::Dynamic> >(residuals.data(),1,residuals.size()) = (((model_params.rotation*src).colwise() + model_params.translation) - dst).colwise().norm();
    return *this;
}

std::vector<float> RigidTransformEstimator::computeResiduals(const RigidTransformParameters &model_params) {
    std::vector<float> residuals;
    computeResiduals(model_params, residuals);
    return residuals;
}

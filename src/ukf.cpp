#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;

  // laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // time when the state is true, in us
  time_us_ = 0.0;

  // state dimension
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // augmented state dimension
  n_aug_ = n_x_ + 2;

  // sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // generate weights vector
  weights_ = VectorXd(2 * n_aug_ + 1);
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  // the current NIS for radar
  NIS_radar_ = 0.0;

  // the current NIS for laser
  NIS_laser_ = 0.0;

  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {


  bool run_radar = (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_);
  bool run_laser = (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_);

  // skip predict/update if sensor type is ignored
  if (!run_radar && !run_laser) {
    return;
  }

  // get alias for measured vector
  VectorXd& z = meas_package.raw_measurements_;

  /*****************************************************************************
  *  Initialization
  ****************************************************************************/

  if (!is_initialized_) {
    if (run_laser) {
      x_(0) = z(0);
      x_(1) = z(1);
    }
    else if (run_radar) {
      // convert radar from polar to cartesian coordinates and compose object state vector
      x_(0) = z(0) * cos(z(1));
      x_(1) = z(0) * sin(z(1));
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;

    // init timestamp
    time_us_ = meas_package.timestamp_;

    return;
  }

  /*****************************************************************************
  *  Prediction
  ****************************************************************************/

  // compute the time elapsed between the current and previous measurements
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Predict(dt);

  /*****************************************************************************
  *  Update
  ****************************************************************************/
  if (run_laser) {
    UpdateLidar(z);
  }
  else if (run_radar) {
    UpdateRadar(z);
  }
}

/**
* Predicts sigma points, the state, and the state covariance matrix.
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
*/
void UKF::Predict(double delta_t) {

  /*****************************************************************************
  *  Generate Augmented Sigma Points
  ****************************************************************************/

  // create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(5) = x_;

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = pow(std_a_, 2);
  P_aug(6, 6) = pow(std_yawdd_, 2);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // fill sigma points matrix with vector as first column
  Xsig_aug.col(0) = x_aug;

  // fill rest sigma points
  MatrixXd L = P_aug.llt().matrixL();
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  /*****************************************************************************
  *  Predict Sigma Points
  ****************************************************************************/

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // extract values for better readability
    double p_x      = Xsig_aug(0, i);
    double p_y      = Xsig_aug(1, i);
    double v        = Xsig_aug(2, i);
    double yaw      = Xsig_aug(3, i);
    double yawd     = Xsig_aug(4, i);
    double nu_a     = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  /*****************************************************************************
  *  Convert Predicted Sigma Points to Mean/Covariance
  ****************************************************************************/

  // predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    x_diff(3) = tools.Normalize(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {VectorXd} z
 */
void UKF::UpdateLidar(VectorXd &z) {

  /*****************************************************************************
  *  UKF Create Measurement Covariance Matrix
  ****************************************************************************/

  // set measurement dimension, lidar can measure p_x and p_y
  int n_z = 2;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);

    // measurement model
    Zsig(0, i) = p_x;
    Zsig(1, i) = p_y;
  }

  // mean predicted measurement
  VectorXd z_pred = GetMeanPredictedMeasurement(n_z, Zsig);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = pow(std_laspx_, 2);
  R(1, 1) = pow(std_laspy_, 2);

  S = S + R;

  /*****************************************************************************
  *  UKF Update for Lidar
  ****************************************************************************/

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  // calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // residual
  VectorXd z_diff = z - z_pred;

  Update(Tc, S, z_diff);

  // calculate NIS
  NIS_laser_ = CalculateNIS(z_diff, S);
}

/**
* Updates the state and the state covariance matrix using a radar measurement.
* @param {VectorXd} z
*/
void UKF::UpdateRadar(VectorXd &z) {

  /*****************************************************************************
  *  UKF Create Measurement Covariance Matrix
  ****************************************************************************/

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                       // r
    Zsig(1, i) = atan2(p_y, p_x);                               // phi
    Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
  }

  // mean predicted measurement
  VectorXd z_pred = GetMeanPredictedMeasurement(n_z, Zsig);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    z_diff(1) = tools.Normalize(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = pow(std_radr_, 2);
  R(1, 1) = pow(std_radphi_, 2);
  R(2, 2) = pow(std_radrd_, 2);

  S = S + R;

  /*****************************************************************************
  *  UKF Update for Radar
  ****************************************************************************/

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  // calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    z_diff(1) = tools.Normalize(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    x_diff(3) = tools.Normalize(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  z_diff(1) = tools.Normalize(z_diff(1));

  // run exact update process
  Update(Tc, S, z_diff);

  // calculate NIS
  NIS_radar_ = CalculateNIS(z_diff, S);
}

/**
 * Generates mean predicted measurement for both laser and radar
 * @param {int} n_z
 * @param {MartixXd} Zsig
 */
VectorXd UKF::GetMeanPredictedMeasurement(int n_z, MatrixXd &Zsig) {
  VectorXd z_pred = VectorXd::Zero(n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  return z_pred;
}

/**
 * Run the exact update process of process matrix and state vector
 * @param {MatrixXd} Tc
 * @param {MatrixXd} S
 * @param {MatrixXd} z_diff
 */
void UKF::Update(MatrixXd &Tc, MatrixXd &S, VectorXd &z_diff) {
  // kalman gain
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}

/**
 * Calculate NIS
 * @param {MatrixXd} Tc
 * @param {MatrixXd} S
 */
double UKF::CalculateNIS(VectorXd &z_diff, MatrixXd &S) {
  return z_diff.transpose() * S.inverse() * z_diff;
}
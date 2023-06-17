// levenberg_marquardt_sinus_fit_01.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <math.h>

# define M_PI 3.14159265358979323846

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

int levmar_sinus(double* t_data_inp, double* y_data_inp, unsigned int M_inp, double* x0_inp, double* x_fit_outp, double k_max_inp, double eps_1_inp, double eps_2_inp, double tau_inp) {
    // convert input 1D arrays to Eigen vectors
    VectorXd t_data_inp_vec(M_inp);
    VectorXd y_data_inp_vec(M_inp);
    // fill input vectors
    for (unsigned int index_0 = 0; index_0 < M_inp; index_0++) {
        t_data_inp_vec(index_0) = t_data_inp[index_0];
        y_data_inp_vec(index_0) = y_data_inp[index_0];
    }
    // convert x0 initial input parameters to Eigen initial vector
    VectorXd x0_inp_vec(3);
    // fill initial parameters vector
    x0_inp_vec(0) = x0_inp[0];
    x0_inp_vec(1) = x0_inp[1];
    x0_inp_vec(2) = x0_inp[2];
    // initial iteration variable
    unsigned int k = 0;
    // auxiliar variable ni
    unsigned int ni = 2;

    // create Jacobian matrix
    MatrixXd J = MatrixXd::Zero(M_inp, 3);
    // fill Jacobian matrix J
    for (unsigned int index_0 = 0; index_0 < M_inp; index_0++){
        // fill 1-st column of Jacobian matrix
        J(index_0, 0) = (-1.0f) * std::sin(t_data_inp_vec(index_0) + x0_inp_vec(1));
        // fill 2-nd column of Jacobian matrix
        J(index_0, 1) = (-1.0f) * x0_inp_vec(0) * std::cos(t_data_inp_vec(index_0) + x0_inp_vec(1));
        // fill 3-rd column of Jacobian matrix
        J(index_0, 2) = -1.0f;
    }
    
    // calculate A matrix
    MatrixXd A = J.transpose() * J;
    // initialize f vector
    VectorXd f(M_inp);
    // calculate f vector
    for (unsigned int index_0 = 0; index_0 < M_inp; index_0++) {
        f(index_0) = y_data_inp_vec(index_0) - x0_inp_vec(0) * std::sin(t_data_inp_vec(index_0) + x0_inp_vec(1)) - x0_inp_vec(2);
    }
    // initialize g vector
    VectorXd g(3);
    g = J.transpose() * f;
    // calculate g norm
    double g_norm = g.norm();
    // define boolean variable
    bool found_bool = (g_norm <= eps_1_inp);
    // define mi variable
    double mi = tau_inp * A.diagonal().maxCoeff();
    // initialize vector x
    VectorXd x(3);
    // fill vector x with x0 initial fitting parameters
    x(0) = x0_inp_vec(0);
    x(1) = x0_inp_vec(1);
    x(2) = x0_inp_vec(2);
    // initilize norm of vector x
    double x_norm = 0.0;

    // initialize vector x_new
    VectorXd x_new(3);
    // fill vector x_new with x0 initial fitting parameters
    x_new(0) = 0.0f;
    x_new(1) = 0.0f;
    x_new(2) = 0.0f;

    // initialize matrix B
    MatrixXd B = MatrixXd::Zero(3, 3);

    // initialize vector h_lm
    VectorXd h_lm(3);
    // initialzie norm of vector h_lm
    double h_lm_norm = 0.0f;

    // define F(x) af F_new(x)
    double F_x = 0.0f;
    double F_x_new = 0.0f;

    //initilaize ro denominator
    double ro_denominator = 0.0f;
    // initilaize ro value
    double ro = 0.0f;

    while (!found_bool && (k < k_max_inp)) {
        //increase iteration number by one
        k++;
        B = A + mi * MatrixXd::Identity(3,3);
        h_lm = (-1.0f) * g.transpose() * B.inverse();
        h_lm_norm = h_lm.norm();
        x_norm = x.norm();
        if (h_lm_norm <= eps_2_inp * (x_norm + eps_2_inp)) {
            found_bool = true;
        }
        else {
            x_new = x + h_lm;
            // calculate F(x)
            // caclulate function f
            for (unsigned int index_0 = 0; index_0 < M_inp; index_0++) {
                f(index_0) = y_data_inp_vec(index_0) - x(0) * std::sin(t_data_inp_vec(index_0) + x(1)) - x(2);
            }
            F_x = 0.5f * f.dot(f);
            for (unsigned int index_0 = 0; index_0 < M_inp; index_0++) {
                f(index_0) = y_data_inp_vec(index_0) - x_new(0) * std::sin(t_data_inp_vec(index_0) + x_new(1)) - x_new(2);
            }
            F_x_new = 0.5f * f.dot(f);
            // ro denominator
            ro_denominator = 0.5f * h_lm.transpose() * (mi * h_lm - g);
            // calculate ro - gain ratio
            ro = (F_x - F_x_new) / ro_denominator;
            if (ro > 0.0f) {
                x = x_new;
                // fill Jacobian matrix
                for (unsigned int index_0 = 0; index_0 < M_inp; index_0++) {
                    // fill 1-st column of Jacobian matrix
                    J(index_0, 0) = (-1.0f) * std::sin(t_data_inp_vec(index_0) + x(1));
                    // fill 2-nd column of Jacobian matrix
                    J(index_0, 1) = (-1.0f) * x(0) * std::cos(t_data_inp_vec(index_0) + x(1));
                    // fill 3-rd column of Jacobian matrix
                    J(index_0, 2) = -1.0f;
                }
                // calculate A matrix
                A = J.transpose() * J;
                // calculate vector f
                for (unsigned int index_0 = 0; index_0 < M_inp; index_0++) {
                    f(index_0) = y_data_inp_vec(index_0) - x(0) * std::sin(t_data_inp_vec(index_0) + x(1)) - x(2);
                }
                // calculate vector g
                g = J.transpose() * f;
                // calculate g norm
                g_norm = g.norm();
                // calculate boolean variable
                found_bool = (g_norm <= eps_1_inp);
                // calculate mi
                mi = mi * std::max(double(1/3), double(1 - pow((2 * ro - 1), 3)));
                // define ni
                ni = 2;
            }
            else {
                // calculate mi
                mi = mi * ni;
                // calculate ni
                ni = 2 * ni;
            }
        }
    }

    //std::cout << x_new << std::endl;
    // convert phase shift from fitting to interval (0, 2*pi)
    if (x_new(1) > 0.0f) {
        x_new(1) = x_new(1) - (2.0f * M_PI) * int(x_new(1) / (2 * M_PI));
    }
    else if (x_new(1) < 0.0f) {
        x_new(1) = x_new(1) - (2.0f * M_PI) * (int(x_new(1) / (2 * M_PI)) - 1);
    }
    else {
        x_new(1) = 0.0f;
    }
    //std::cout << x_new << std::endl;

    // store fitting results to output 1D double array
    x_fit_outp[0] = x_new(0);
    x_fit_outp[1] = x_new(1);
    x_fit_outp[2] = x_new(2);

    return 0;
}

int main()
{
    // define experimental sinus parameters
    // experimental amplitude
    double x1 = 123.0f;
    // experimental phase shift
    double x2 = M_PI / 4.0f;
    // experimental offset
    double x3 = 17.0f;
    // define experimental sinus parameter vector
    double* x = new double[3];
    // fill experimental parameters vector
    x[0] = x1;
    x[1] = x2;
    x[2] = x3;
    // define intial parameters of sinusoidal function
    // intial amplitude
    double x01 = 1.0f;
    // initial phase shift
    double x02 = M_PI/2.0f;
    // initial offset
    double x03 = 0.0f;
    // define initial parameter vector
    double* x0 = new double[3];
    // fill initial parameters vector
    x0[0] = x01;
    x0[1] = x02;
    x0[2] = x03;
    // define number of steps in fringe scanning
    // define number of experimental points
    const unsigned int M = 5;
    // define input t_data 1D array, on the t axis 
    double* t_data = new double[M];
    for (unsigned int index_0 = 0; index_0 < M; index_0++) {
        t_data[index_0] = double(index_0) * ((2.0f * M_PI) / double(M));
    }
    // define input y_data 1D array
    double* y_data = new double[M];
    for (unsigned int index_0 = 0; index_0 < M; index_0++) {
        y_data[index_0] = x[0] * std::sin(t_data[index_0] + x[1]) + x[2];
    }
    // maximum number of iterations in fitting
    unsigned int k_max = 1000;
    // auxiliar fitting variable epsilon 1
    double eps_1 = 1.0E-8;
    // auxiliar fitting variable epsilon 2
    double eps_2 = 1.0E-8;
    // auxiliar fitting variable tau
    double tau = 1.0E-3;
    // create output 1D array where fitting results will be stored
    double* x_fit = new double[3];

    // perform Levenberg-Marquardt sinusoidal fitting
    levmar_sinus(t_data, y_data, M, x0, x_fit,  k_max, eps_1, eps_2, tau);

    // print fitting results on the screen
    std::cout << "The amplitude of sinus curve is: " << x_fit[0] << std::endl;
    std::cout << "The phase of sinus curve is: " << x_fit[1] << std::endl;
    std::cout << "The offset of sinus curve is: " << x_fit[2] << std::endl;

    // delete buffers
    delete[] x;
    delete[] x0;
    delete[] t_data;
    delete[] y_data;
    delete[] x_fit;

    return 0;
}

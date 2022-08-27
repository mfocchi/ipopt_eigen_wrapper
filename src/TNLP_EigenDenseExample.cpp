// Copyright (C) 2014, LAAS-CNRS
//
// Author: Andrea Del Prete     LAAS-CNRS    2014-06-24

#include "TNLP_EigenDenseExample.hpp"

#include <cassert>
#include <iostream>


using namespace Eigen;
using namespace std;

/**************************************************************************
 **************************************************************************
                        TNLP_EigenDenseExample
 **************************************************************************
 ***************************************************************************/
TNLP_EigenDenseExample::TNLP_EigenDenseExample()
{
}

/** default destructor */
TNLP_EigenDenseExample::~TNLP_EigenDenseExample()
{
}

bool TNLP_EigenDenseExample::get_nlp_info(int &n, int &m)
{
    // The problem described in HS071_NLP.hpp has 4 variables, x[0] through x[3]
    n = 4;
    
    // one equality constraint and one inequality constraint
    m = 2;
    
    return true;
}

/** Method to return the bounds for my problem */
bool TNLP_EigenDenseExample::get_bounds_info(EVector x_l, EVector x_u,
                                             EVector g_l, EVector g_u)
{
    // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
    // If desired, we could assert to make sure they are what we think they are.
    assert(x_l.size() == 4);
    assert(g_l.size() == 2);
    
    // the variables have lower bounds of 1
    for (Ipopt::Index i=0; i<4; i++)
        x_l[i] = 1.0;
    
    // the variables have upper bounds of 5
    for (Ipopt::Index i=0; i<4; i++)
        x_u[i] = 5.0;
    
    // the first constraint g1 has a lower bound of 25
    g_l[0] = 25;
    // the first constraint g1 has NO upper bound, here we set it to 2e19.
    // Ipopt interprets any number greater than nlp_upper_bound_inf as
    // infinity. The default value of nlp_upper_bound_inf and nlp_lower_bound_inf
    // is 1e19 and can be changed through ipopt options.
    g_u[0] = 2e19;
    
    // the second constraint g2 is an equality constraint, so we set the
    // upper and lower bound to the same value
    g_l[1] = g_u[1] = 40.0;
    
    return true;
}

/** Method to return the starting point for the algorithm */
bool TNLP_EigenDenseExample::get_starting_point(bool init_x, EVector x,
                                              bool init_z, EVector z_L, EVector z_U,
                                              bool init_lambda, EVector lambda)
{
    // Here, we assume we only have starting values for x, if you code
    // your own NLP, you can provide starting values for the dual variables
    // if you wish
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);
    
    // initialize to the given starting point
    x[0] = 1.0;
    x[1] = 5.0;
    x[2] = 5.0;
    x[3] = 1.0;
    
    return true;
}

/** Method to return the objective value */
//bool DummyTNLP_EigenDense::eval_f(int n, EConstVector &x, bool new_x, double &obj_value) = 0;
bool TNLP_EigenDenseExample::eval_f(EConstVector x, bool new_x, double &obj_value)
{
    assert(x.size() == 4);
    obj_value = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    return true;
}

/** Method to return the gradient of the objective */
bool TNLP_EigenDenseExample::eval_grad_f(EConstVector x, bool new_x, EVector grad_f)
{
    assert(x.size() == 4);
    
    grad_f[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
    grad_f[1] = x[0] * x[3];
    grad_f[2] = x[0] * x[3] + 1;
    grad_f[3] = x[0] * (x[0] + x[1] + x[2]);
    
    return true;
}

/** Method to return the constraint residuals */
bool TNLP_EigenDenseExample::eval_g(EConstVector x, bool new_x, EVector g)
{
    assert(x.size() == 4);
    assert(g.size() == 2);
    
    g[0] = x[0] * x[1] * x[2] * x[3];
    g[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
    
    return true;
}

/** Method to return the values of the Jacobian. */
bool TNLP_EigenDenseExample::eval_jac_g(EConstVector x, bool new_x,
                                      EMatrix values)
{
    // return the values of the jacobian of the constraints
    values(0,0) = x[1]*x[2]*x[3]; // 0,0
    values(0,1) = x[0]*x[2]*x[3]; // 0,1
    values(0,2) = x[0]*x[1]*x[3]; // 0,2
    values(0,3) = x[0]*x[1]*x[2]; // 0,3
    
    values(1,0) = 2*x[0]; // 1,0
    values(1,1) = 2*x[1]; // 1,1
    values(1,2) = 2*x[2]; // 1,2
    values(1,3) = 2*x[3]; // 1,3
    
    return true;
}

/** Method to return the values of the hessian of the lagrangian. */
bool TNLP_EigenDenseExample::eval_h(EConstVector x, bool new_x,
                                  double obj_factor, EConstVector lambda,
                                  bool new_lambda, EMatrix values)
{
    // return the values. This is a symmetric matrix, fill the lower left
    // triangle only
    
    // fill the objective portion
    values(0,0) = obj_factor * (2*x[3]); // 0,0
    
    values(1,0) = obj_factor * (x[3]);   // 1,0
    values(1,1) = 0.;                    // 1,1
    
    values(2,0) = obj_factor * (x[3]);   // 2,0
    values(2,1) = 0.;                    // 2,1
    values(2,2) = 0.;                    // 2,2
    
    values(3,0) = obj_factor * (2*x[0] + x[1] + x[2]); // 3,0
    values(3,1) = obj_factor * (x[0]);                 // 3,1
    values(3,2) = obj_factor * (x[0]);                 // 3,2
    values(3,3) = 0.;                                  // 3,3
    
    
    // add the portion for the first constraint
    values(1,0) += lambda[0] * (x[2] * x[3]); // 1,0
    
    values(2,0) += lambda[0] * (x[1] * x[3]); // 2,0
    values(2,1) += lambda[0] * (x[0] * x[3]); // 2,1
    
    values(3,0) += lambda[0] * (x[1] * x[2]); // 3,0
    values(3,1) += lambda[0] * (x[0] * x[2]); // 3,1
    values(3,2) += lambda[0] * (x[0] * x[1]); // 3,2
    
    // add the portion for the second constraint
    values(0,0) += lambda[1] * 2; // 0,0
    values(1,1) += lambda[1] * 2; // 1,1
    values(2,2) += lambda[1] * 2; // 2,2
    values(3,3) += lambda[1] * 2; // 3,3
    
    values.triangularView<Upper>() = values.triangularView<Lower>().transpose();

    return true;
}

void TNLP_EigenDenseExample::finalize_solution(Ipopt::SolverReturn status, EConstVector x,
                                             EConstVector z_L, EConstVector z_U,
                                             EConstVector g, EConstVector lambda,
                                             double obj_value, const Ipopt::IpoptData* ip_data,
                                             Ipopt::IpoptCalculatedQuantities* ip_cq)
{
    // here is where we would store the solution to variables, or write to a file, etc
    // so we could use the solution.
    
    // For this example, we write the solution to the console
    std::cout << std::endl << std::endl << "Solution of the primal variables, x" << std::endl;
    std::cout << "x = " << x << std::endl;
    
    std::cout << std::endl << std::endl << "Solution of the bound multipliers, z_L and z_U" << std::endl;
    std::cout << "z_L = " << z_L << std::endl;
    std::cout << "z_U = " << z_U << std::endl;
    
    std::cout << std::endl << std::endl << "Objective value" << std::endl;
    std::cout << "f(x*) = " << obj_value << std::endl;
    
    std::cout << std::endl << "Final value of the constraints:" << std::endl;
    std::cout << "g = " << g << std::endl;
}


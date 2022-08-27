// Copyright (C) 2014, LAAS-CNRS
//
// Author: Andrea Del Prete     LAAS-CNRS    2014-06-24

#ifndef __TNLP_EIGEN_DENSE_EXAMPLE_HPP__
#define __TNLP_EIGEN_DENSE_EXAMPLE_HPP__

#include "TNLP_EigenDense.hpp"
#include <Eigen/Core>
#define prt(x) std::cout << #x " = \n" << x << "\n" << std::endl;


class TNLP_EigenDenseExample : public TNLP_EigenDense
{
public:
    /** default constructor */
    TNLP_EigenDenseExample();
    
    /** default destructor */
    virtual ~TNLP_EigenDenseExample();
    
    virtual bool get_nlp_info(int &n, int &m);
    
    /** Method to return the bounds for my problem */
    virtual bool get_bounds_info(EVector x_l, EVector x_u,
                                 EVector g_l, EVector g_u);
    
    /** Method to return the starting point for the algorithm */
    virtual bool get_starting_point(bool init_x, EVector x,
                                    bool init_z, EVector z_L, EVector z_U,
                                    bool init_lambda, EVector lambda);
    
    /** Method to return the objective value */
    virtual bool eval_f(EConstVector x, bool new_x, double &obj_value);
    
    /** Method to return the gradient of the objective */
    virtual bool eval_grad_f(EConstVector x, bool new_x, EVector grad_f);
    
    /** Method to return the constraint residuals */
    virtual bool eval_g(EConstVector x, bool new_x, EVector g);
    
    /** Method to return the values of the Jacobian. */
    virtual bool eval_jac_g(EConstVector x, bool new_x,
                            EMatrix values);
    
    /** Method to return the values of the hessian of the lagrangian. */
    virtual bool eval_h(EConstVector x, bool new_x,
                        double obj_factor, EConstVector lambda,
                        bool new_lambda, EMatrix values);
    
    //@}
    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    virtual void finalize_solution(Ipopt::SolverReturn status, EConstVector x,
                                   EConstVector z_L, EConstVector z_U,
                                   EConstVector g, EConstVector lambda,
                                   double obj_value, const Ipopt::IpoptData* ip_data,
                                   Ipopt::IpoptCalculatedQuantities* ip_cq);
    
private:
    //  TNLP_EigenDenseExample();
    TNLP_EigenDenseExample(const TNLP_EigenDenseExample&);
    TNLP_EigenDenseExample& operator=(const TNLP_EigenDenseExample&);
};


#endif

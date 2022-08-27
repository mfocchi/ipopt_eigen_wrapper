// Copyright (C) 2014, LAAS-CNRS
//
// Author: Andrea Del Prete     LAAS-CNRS    2014-06-24
#define HAVE_CSTDDEF
#include <IpTNLP.hpp>
#undef HAVE_CSTDDEF
#include "IpIpoptApplication.hpp"
#include "TNLP_EigenDenseExample.hpp"
#include <hs071_nlp.hpp>
#include <iostream>

//Timer
 #include <sys/time.h>

using namespace Ipopt;

const bool TEST_EIGEN_WRAPPER = true;

int main(int argv, char* argc[])
{
  // Create a new instance of your nlp
  //  (use a SmartPtr, not raw)
    SmartPtr<TNLP> mynlp;
    if(TEST_EIGEN_WRAPPER)
        mynlp = new TNLP_EigenDenseExample();
    else
        mynlp = new HS071_NLP();

  // Create a new instance of IpoptApplication
  //  (use a SmartPtr, not raw)
  // We are using the factory, since this allows us to compile this
  // example with an Ipopt Windows DLL
  SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
  app->RethrowNonIpoptException(true);

  // Change some options
  // Note: The following choices are only examples, they might not be
  //       suitable for your optimization problem.
  app->Options()->SetNumericValue("tol", 1e-7);
  app->Options()->SetStringValue("mu_strategy", "adaptive");
  app->Options()->SetStringValue("output_file", "ipopt.out");

  app->Options()->SetIntegerValue("max_iter", (int)100);

 // app->Options()->SetIntegerValue("max_iter", (int)1);
  app->Options()->SetIntegerValue("print_level", (int)5);//verbosity
  //app->Options()->SetStringValue("jacobian_approximation","finite-difference-values");// compute jacobian numerically(should set return true in the grad functions!)
  app->Options()->SetStringValue("hessian_approximation","limited-memory");// compute hessian numerically
  //app->Options()->SetStringValue("derivative_test","first-order");//
  //app->Options()->SetStringValue("derivative_test_print_all","yes");//


  // The following overwrites the default name (ipopt.opt) of the
  // options file
  // app->Options()->SetStringValue("option_file_name", "hs071.opt");

  // Initialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = app->Initialize();
  if (status != Solve_Succeeded) {
    std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
    return (int) status;
  }

  // Ask Ipopt to solve the problem
  status = app->OptimizeTNLP(mynlp);

  if (status == Solve_Succeeded) {
    std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
  }
  else {
    std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
  }

  // As the SmartPtrs go out of scope, the reference count
  // will be decremented and the objects will automatically
  // be deleted.

  return (int) status;
}




#include "SingleEdgeOptimizer.h"
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/emst/dtb.hpp>
#include <iostream>
#include <math.h>
#include <vector>
#include "ExponentialDensity.h"
#include "LinearDensity.h"
#include "FunctionEvaluator.h"
#include <mlpack/core.hpp>
#include "DensityComputator.h"
#include "GeneralConvertor.h"
#include "NaiveMethods.h"
#include "DensityEstimationRules.h"
#include "TreeExperements.h"
#include <mlpack/core/tree/statistic.hpp>
#include "AMSTComputator.h"
#include <ctime>
#include "TestPerformer.h"
#include <cstdlib>

SingleEdgeOptimizer::SingleEdgeOptimizer()
{
    //ctor
}


void SingleEdgeOptimizer::Gradient(arma::vec p, arma::vec u, arma::vec v, arma::mat & finale, std::ofstream & mystream)
{

    double tvalue = dot(p-v,p-u) / dot(u-v,u-v);
    double d[3];
    for (int i = 0; i < 3; i++)
    {
        d[i] = (u[i] - v[i])*(u[i]-v[i]);
    }

    if (tvalue < 0)
    {
        finale = arma::mat(1,6);
        for (int i = 0; i < 3; i++)
        {
            finale(0,i) = -2*p(i) + 2*u(i);
        }
        for (int i = 3; i < 6; i++)
        {
            finale(0,i) = 0;

        }
        mystream << "Case 1 detected" << std::endl;
    }
    else if (tvalue > 1)
    {
        finale = arma::mat(1,6);
        for (int i = 0; i < 3; i++)
        {
            finale(0,i) = 0;
        }


        for (int i = 3; i < 6; i++)
        {
            finale(0,i) = -2*p(i-3) + 2*u(i-3);
        }
        mystream << "Case 2 detected" << std::endl;
    }
    else
    {
        arma::vec t(6);
        t(0) = -(p(0) - v(0))/( d[0] + d[1] + d[2]) - 2*((p(0) - u(0))*(p(0) - v(0)) + (p(1) - u(1))*(p(1) - v(1)) + (p(2) - u(2))*(p(2) - v(2)))*(u(0) - v(0))/((d[0] + d[1] + d[2])*(d[0] + d[1] + d[2]));
        t(1) = -(p(1) - v(1))/(d[0] + d[1] + d[2]) - 2*((p(0) - u(0))*(p(0)- v(0)) + (p(1) - u(1))*(p(1) - v(1)) + (p(2) - u(2))*(p(2) - v(2)))*(u(1) - v(1))/((d[0] + d[1] + d[2])*(d[0] + d[1] + d[2]));
        t(2) = -(p(2) - v(2))/(d[0] + d[1] + d[2]) - 2*((p(0) - u(0))*(p(0)- v(0)) + (p(1) - u(1))*(p(1) - v(1)) + (p(2) - u(2))*(p(2) - v(2)))*(u(2) - v(2))/((d[0] + d[1] + d[2])*(d[0] + d[1] + d[2]));
        t(3) = -(p(0) - u(0))/( d[0] + d[1] + d[2]) + 2*((p(0)- u(0))*(p(0) - v(0)) + (p(1) - u(1))*(p(1) - v(1)) + (p(2) - u(2))*(p(2) - v(2)))*(u(0) - v(0))/((d[0] + d[1] + d[2])*(d[0] + d[1] + d[2]));
        t(4) = -(p(1) - u(1))/( d[0] + d[1] + d[2]) + 2*((p(0)- u(0))*(p(0) - v(0)) + (p(1) - u(1))*(p(1) - v(1)) + (p(2) - u(2))*(p(2) - v(2)))*(u(1) - v(1))/((d[0] + d[1] + d[2])*(d[0] + d[1] + d[2]));
        t(5) = -(p(2) - u(2))/( d[0] + d[1] + d[2]) + 2*((p(0)- u(0))*(p(0) - v(0)) + (p(1) - u(1))*(p(1) - v(1)) + (p(2) - u(2))*(p(2) - v(2)))*(u(2) - v(2))/((d[0] + d[1] + d[2])*(d[0] + d[1] + d[2]));
        arma::mat tm(7,6);
        for (int i = 0; i < 7; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                if (i != 6)
                {
                    if (i == j)
                    {
                        tm(i,j) = 1;
                    }
                    else
                    {
                        tm(i,j) = 0;
                    }
                }
                else
                {
                    tm(i,j) = t(j);
                }

            }
        }
        arma::mat ww(1,7);
        for (int i = 0; i < 3; i++)
        {
            ww(0,i) = 2*(tvalue*u(i) - (tvalue -1)*v(i) -p(i))*tvalue;
        }
        // ww(0,i) = 2*(tvalue*u(i) - (tvalue - 1)*v(i) – p(i))*tvalue;
        for (int i = 3; i < 6; i++)
        {
            // ww(0,i) = -2*(tvalue*u(i-3) -(tvalue - 1)*v(i-3) - p(i-3))*(tvalue – 1),
            ww(0,i) = -2*(tvalue*u(i-3) -(tvalue - 1)*v(i-3) - p(i-3))*(tvalue - 1);
            // ww(0,i) = -2*(tvalue*u(i-3) -(tvalue - 1)*v(i-3) - p(i-3))*(tvalue - 1);
        }
        ww(0,6) = 2*(tvalue*u(0) - (tvalue - 1)*v(0) - p(0))*(u(0) - v(0));
        ww(0,6) = ww(0,6) + 2*(tvalue*u(1) - (tvalue - 1)*v(1) - p(1))*(u(1) - v(1));
        ww(0,6) = ww(0,6) + 2*(tvalue*u(2) - (tvalue - 1)*v(2) - p(2))*(u(2) -v(2));
        finale = ww*tm;
        mystream << "Case 3 detected" << std::endl;
    }
}


double Evaluate(arma::vec & p, arma::vec & u, arma::vec & v)
{
    double t = dot(p-v,p-u) / dot(u-v,u-v);
    arma::vec bambam = t*u+(1-t)*v;
    return mlpack::metric::EuclideanDistance::Evaluate(p,bambam);

}

double EvaluateOnPointCloud( arma::vec & u, arma::vec & v, arma::mat & cloud )
{
    double sum = 0;
    for (int i = 0; i < cloud.n_cols; i++)
    {
        arma::vec p = cloud.col(i);
        sum = sum + Evaluate(p,u,v);
    }
    return sum;
}
void SingleEdgeOptimizer::EvaluateDifferentialOnPointCloud( arma::vec & u, arma::vec & v, arma::mat & cloud, arma::vec & ur, arma::vec & vr,  std::ofstream & mystream)
{
    arma::mat rur(1,6);
    rur.zeros();

    for (int i = 0; i < cloud.n_cols; i++)
    {
        mystream << "Round " << i << " starting!" << std::endl;
        arma::mat utmp(1,6);
        utmp.zeros();
        arma::vec p = cloud.col(i);
        //  std::cout << "Number: " << i << std::endl;
        //std::cout << p << std::endl;
        SingleEdgeOptimizer::Gradient(p, u, v, utmp,mystream);
        mystream << utmp << std::endl;
        rur = rur + utmp;
    }
    for (int i = 0; i < 2; i++)
    {
        ur(i) = rur(0,i);
    }
    for (int i = 3; i < 6; i++)
    {
        vr(i-3) = rur(0,i);
    }


}

double distanceBetweenConfigurations(arma::vec & u, arma::vec & v, arma::vec u2, arma::vec v2)
{

    return std::max(mlpack::metric::EuclideanDistance::Evaluate(u,u2),mlpack::metric::EuclideanDistance::Evaluate(v,v2));

}

void SingleEdgeOptimizer::SimpleFunctionMinimizer(double gamma, int max_iter, double precision, arma::vec & u, arma::vec & v, arma::mat & cloud)
{

//! For debugging we need this file:

    std::ofstream mystream;
    mystream.open("/home/yury/Dropbox/Github/DataSkeletonizationNew/outputs/EdgeTest/First_Test/debug.txt");
//! We implement this using constant linear search algorithm


    int iter = 0;
    double previousStepSize = 100.0;
    while((iter < max_iter) && (previousStepSize > precision))
    {
        mystream << "Round " << iter << std::endl;
        arma::vec u1(3);
        arma::vec v1(3);
        arma::vec u2(3);
        arma::vec v2(3);
        EvaluateDifferentialOnPointCloud(u, v, cloud, u2, v2,mystream);
        std::cout << "Vector U:" << std::endl;
        std::cout << u << std::endl;
        std::cout << "Vector V:" << std::endl;
        std::cout << v << std::endl;
        std::cout << "Vector U2: " << std::endl;
        std::cout << u2 << std::endl;
        std::cout << "Vector V2: " << std::endl;
        std::cout << v2 << std::endl;
        u1 =  u - gamma*u2;
        v1 =  v - gamma*v2;
        std::cout << "Vector U1: " << std::endl;
        std::cout << u1 << std::endl;
        std::cout << "Vector V1: " << std::endl;
        std::cout << v1 << std::endl;
        double dconfig = distanceBetweenConfigurations(u,v,u1,v1);
        u = u1;
        v = v1;
        previousStepSize = dconfig;
        iter++;
    }
    mystream.close();



}



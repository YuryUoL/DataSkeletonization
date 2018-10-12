#ifndef SINGLEEDGEOPTIMIZER_H
#define SINGLEEDGEOPTIMIZER_H
#include <mlpack/core.hpp>


class SingleEdgeOptimizer
{
    public:
        SingleEdgeOptimizer();
        static void Gradient(arma::vec p, arma::vec u, arma::vec v, arma::mat & finale,std::ofstream & mystream);
        static void EvaluateDifferentialOnPointCloud( arma::vec & u, arma::vec & v, arma::mat & cloud, arma::vec & ur, arma::vec & vr,std::ofstream & mystream );
        //! The main player of this team is:
        static void SimpleFunctionMinimizer(double gamma, int max_iter, double precision, arma::vec & u, arma::vec & v,arma::mat & cloud);

    protected:

    private:
};

#endif // SINGLEEDGEOPTIMIZER_H

#ifndef GRADIENTDESCENDTESTER_H
#define GRADIENTDESCENDTESTER_H

#include "SingleEdgeOptimizer.h"
#include "CloudGenerator.h"
#include "SingleEdgeOptimizer.h"

class GradientDescendTester
{
public:
    GradientDescendTester() {}
    static void QuickTest(std::string folder)
    {
        //! We generate points [0,100] interval
        MyGraphType G;
        GraphGeneration::SimpleEdgeGraph(G,100);
        arma::mat datat;
        CloudGenerator::generatePointsOnGraphInArmaMat(G, 1000,5,datat);
        arma::mat data = datat.t();
        //! We set our first guess to be the diagonal
        arma::vec u{5,5,5};
        arma::vec v{95,-5,-5};
        double gamma = pow(10,-4);
        int maxitr = 1000;
        double minError = 1.0;

        std::cout << "Still allright" << std::endl;


        SingleEdgeOptimizer::SimpleFunctionMinimizer(gamma, maxitr, minError,u,v,data);
        std::cout << "Resuluts: " << std::endl;
        std::cout << "Vector 1: " << std::endl << u << std::endl;
        std::cout << "Vector 2: " << std::endl << v << std::endl;
        std::string cloudpath= folder + "cloud.vtk";
        std::string edgepath = folder + "edge.vtk";
        MyGraphType EdgeGraph;
        vertex_descriptor uedge = Graph::add_vertex(EdgeGraph, Point(u(0), u(1), u(2)));
        vertex_descriptor vedge = Graph::add_vertex(EdgeGraph, Point(v(0), v(1), v(2)));
        Graph::add_edge(EdgeGraph,uedge, vedge);
        GeneralConvertor::GraphToVtk(edgepath,EdgeGraph);

        //! Here wer just compute classical minimum spanning tree so it would be easier to plot the point cloud

        mlpack::emst::DualTreeBoruvka<> MSTOP(data);
        arma::mat results;
        MSTOP.ComputeMST(results);
        GeneralConvertor::MSTToVTK(data,results, cloudpath);

    }

protected:

private:
};

#endif // GRADIENTDESCENDTESTER_H

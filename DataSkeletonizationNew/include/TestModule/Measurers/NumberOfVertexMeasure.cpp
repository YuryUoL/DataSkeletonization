#include "NumberOfVertexMeasure.h"

NumberOfVertexMeasure::NumberOfVertexMeasure(int precision):
    AbstractMeasurer<int>("Number of vertices", precision)
{
    //ctor
}

int NumberOfVertexMeasure::CompleteMeasurments(MyGraphType & G,generatable* cloud, std::list<Point>* generatedCloud)
{
    return boost::num_vertices(G);
}
std::string NumberOfVertexMeasure::returnStatisticString()
{
    double avg = mystatistic.returnAvg();
    std::stringstream stream;
    stream << std::fixed << std::setprecision(precision) << avg;
    std::string s = stream.str();
    return s;
}

//NumberOfVertexMeasure::~NumberOfVertexMeasure()
//{
    //dtor
//}

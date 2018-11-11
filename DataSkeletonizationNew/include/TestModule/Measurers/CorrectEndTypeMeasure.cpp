#include "CorrectEndTypeMeasure.h"
#include "Graph.h"


CorrectEndTypeMeasure::CorrectEndTypeMeasure(int precision):SimpleMeasurerCore("Endpoints",precision)
{
    //ctor
}


bool CorrectEndTypeMeasure::CompleteMeasurments(MyGraphType & G,generatable* cloud, std::list<Point>* generatedCloud)
{
    bool theVariable = cloud->DoesGraphHaveCorrectForm(G);
   // std::cout << "The descision (does graph have the correct form? " << theVariable << std::endl;
    return theVariable;
}






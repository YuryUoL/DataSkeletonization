#ifndef NUMBEROFVERTEXMEASURE_H
#define NUMBEROFVERTEXMEASURE_H

#include "Graph.h"
#include "AbstractMeasurer.h"

class NumberOfVertexMeasure : public AbstractMeasurer<int>
{
public:
    NumberOfVertexMeasure(int precision);
    int CompleteMeasurments(MyGraphType & G,generatable* cloud, std::list<Point>* generatedCloud) override;
    std::string returnStatisticString() override;
    NumberOfVertexMeasure* Clone() override
    {
        return new NumberOfVertexMeasure(*this);
    }
    //virtual ~NumberOfVertexMeasure();

protected:

private:
};

#endif // NUMBEROFVERTEXMEASURE_H

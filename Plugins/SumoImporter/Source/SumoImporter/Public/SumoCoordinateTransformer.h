#pragma once

#include "CoreMinimal.h"
#include "SumoTypes.h"

class SUMOIMPORTER_API FSumoCoordinateTransformer
{
public:
	static FVector TransformPointMeters(const FVector& SumoPointMeters, const FSumoTransformConfig& Config);
	static FVector TransformDirectionMeters(const FVector& SumoDirectionMeters, const FSumoTransformConfig& Config);
	static void TransformPointsMeters(const TArray<FVector>& InPointsMeters, const FSumoTransformConfig& Config, TArray<FVector>& OutPointsCm);

private:
	static FVector ApplyAxisMapping(const FVector& InPoint, ESumoAxisMapping Mapping);
};

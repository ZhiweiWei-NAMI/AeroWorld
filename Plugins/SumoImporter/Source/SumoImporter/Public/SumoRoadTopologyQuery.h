#pragma once

#include "CoreMinimal.h"
#include "SumoTypes.h"

class UWorld;

class SUMOIMPORTER_API FSumoRoadTopologyQuery
{
public:
	static bool FindNearestRoadSample(
		UWorld* World,
		const FVector& QueryWorldCm,
		FSumoNearestLaneSample& OutSample,
		FString& OutError);
};

#pragma once

#include "CoreMinimal.h"
#include "SumoTypes.h"

class ASumoRoadNetworkActor;
class UWorld;

class SUMOIMPORTER_API FSumoSceneBuilder
{
public:
	static ASumoRoadNetworkActor* BuildToWorld(
		UWorld* World,
		const FSumoNetData& NetData,
		const FSumoTransformConfig& TransformConfig,
		const FSumoBuildOptions& BuildOptions,
		FSumoImportStats& InOutStats,
		FString& OutError);
};
